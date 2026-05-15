//
//  Sky.metal
//  Pocketcat
//
//  Created by Amélie Heinrich on 03/04/2026.
//

#include "Common/Math.h"
#include "Common/Bindless.h"

#include <metal_stdlib>
using namespace metal;

struct atmosphere_parameters {
    float4 rayleigh_scattering;       // xyz = coefficients, w = unused
    float rayleigh_scale_height;
    float rayleight_absorption_base;
    float mie_scattering;
    float mie_absorption;
    float mie_scale_height;
    float mie_phase_g;
    float2 _pad0;                     // explicit padding to align next float4
    float4 ozone_absorption;          // xyz = coefficients, w = unused
    float4 ground_albedo;             // xyz = albedo, w = unused
    float ground_radius_mm;
    float atmosphere_radius_mm;
    float sky_intensity;
    float sun_disk_intensity;         // brightness of the fake sun disk
    float sun_disk_size;              // angular diameter in degrees
    float night_blend_factor;         // 0=full night, 1=full day (clamped blend zone)
    float milkyway_exposure;
};

constant float sun_transmittance_steps = 40.0f;
constant float mul_scatt_steps = 20.0f;
constant int sqrt_samples = 8;
constant int num_scattering_steps = 32;

inline void get_scattering_values(float3 pos,
                                  thread float3 &rayleight_scattering,
                                  thread float  &mie_scattering,
                                  thread float3 &extinction,
                                  const device atmosphere_parameters& parameters) {
    float altitude_km = (length(pos) - parameters.ground_radius_mm) * 1000.0f;
    float rayleight_density = exp(-altitude_km / parameters.rayleigh_scale_height);
    float mie_density = exp(-altitude_km / parameters.mie_scale_height);

    rayleight_scattering = parameters.rayleigh_scattering.xyz * rayleight_density;
    float rayleigh_absorption = parameters.rayleight_absorption_base * rayleight_density;

    mie_scattering = parameters.mie_scattering * mie_density;
    float mie_absorption = parameters.mie_absorption * mie_density;

    float3 ozone_absorption = parameters.ozone_absorption.xyz * max(0.0f, 1.0f - abs(altitude_km - 25.0f) / 15.0f);

    extinction = rayleight_scattering + rayleigh_absorption + mie_scattering + mie_absorption + ozone_absorption;
}

static float3 get_sun_transmittance(float3 pos, float3 sun_dir, const device atmosphere_parameters& parameters) {
    if (ray_intersect_sphere(pos, sun_dir, parameters.ground_radius_mm - 0.001f) > 0.0f)
        return float3(0.0f);

    float atmo_dist = ray_intersect_sphere(pos, sun_dir, parameters.atmosphere_radius_mm);
    float t = 0.0f;
    float3 transmittance = float3(1.0f);

    for (float i = 0.0f; i < sun_transmittance_steps; i += 1.0f) {
        float new_t = ((i + 0.3f) / sun_transmittance_steps) * atmo_dist;
        float dt = new_t - t;
        t = new_t;

        float3 new_pos = pos + t * sun_dir;
        float3 rayleigh_scattering, extinction;
        float mie_scattering;
        get_scattering_values(new_pos, rayleigh_scattering, mie_scattering, extinction, parameters);
        transmittance *= exp(-dt * extinction);
    }
    return transmittance;
}

inline float3 get_val_from_tlut(texture2d<float> tex, float3 pos, float3 sun_dir, const device atmosphere_parameters& parameters) {
    constexpr sampler s(filter::linear, address::clamp_to_edge, coord::normalized);
    float height = length(pos);
    float3 up = pos / height;
    float sun_cos_zenith = dot(sun_dir, up);
    float2 uv = float2(
        clamp(0.5f + 0.5f * sun_cos_zenith, 0.0f, 1.0f),
        max(0.0f, min(1.0f, (height - parameters.ground_radius_mm) / (parameters.atmosphere_radius_mm - parameters.ground_radius_mm)))
    );
    return tex.sample(s, uv).rgb;
}

inline float3 get_val_from_mslut(texture2d<float> tex, float3 pos, float3 sun_dir, const device atmosphere_parameters& parameters) {
    constexpr sampler s(filter::linear, address::clamp_to_edge, coord::normalized);
    float height = length(pos);
    float3 up = pos / height;
    float sun_cos_zenith = dot(sun_dir, up);
    float2 uv = float2(
        clamp(0.5f + 0.5f * sun_cos_zenith, 0.0f, 1.0f),
        max(0.0f, min(1.0f, (height - parameters.ground_radius_mm) / (parameters.atmosphere_radius_mm - parameters.ground_radius_mm)))
    );
    return tex.sample(s, uv).rgb;
}

float3 get_spherical_dir(float theta, float phi) {
    float cos_phi = cos(phi), sin_phi = sin(phi);
    float cos_theta = cos(theta), sin_theta = sin(theta);
    return float3(sin_phi * sin_theta, cos_phi, sin_phi * cos_theta);
}

void get_mul_scatt_values(float3 pos,
                          float3 sun_dir,
                          texture2d<float> tlut,
                          thread float3 &lum_total,
                          thread float3 &fms,
                          const device atmosphere_parameters& parameters)
{
    lum_total = float3(0.0f);
    fms = float3(0.0f);

    float inv_samples = 1.0f / float(sqrt_samples * sqrt_samples);

    for (int i = 0; i < sqrt_samples; i++) {
        for (int j = 0; j < sqrt_samples; j++) {
            float theta = M_PI_F * (float(i) + 0.5f) / float(sqrt_samples);
            float phi = safeacos(1.0f - 2.0f * (float(j) + 0.5f) / float(sqrt_samples));
            float3 ray_dir = get_spherical_dir(theta, phi);

            float atmo_dist = ray_intersect_sphere(pos, ray_dir, parameters.atmosphere_radius_mm);
            float ground_dist = ray_intersect_sphere(pos, ray_dir, parameters.ground_radius_mm);
            float t_max = (ground_dist > 0.0f) ? ground_dist : atmo_dist;

            float cos_theta = dot(ray_dir, sun_dir);
            float mie_phase_value = get_mie_phase(cos_theta);
            float rayleigh_phase = get_rayleigh_phase(-cos_theta);

            float3 lum = float3(0.0f), lum_factor = float3(0.0f), transmittance = float3(1.0f);
            float t = 0.0f;

            for (float step_i = 0.0f; step_i < mul_scatt_steps; step_i += 1.0f) {
                float new_t = ((step_i + 0.3f) / mul_scatt_steps) * t_max;
                float dt = new_t - t;
                t = new_t;

                float3 new_pos = pos + t * ray_dir;
                float3 rayleigh_scattering, extinction;
                float mie_scattering;
                get_scattering_values(new_pos, rayleigh_scattering, mie_scattering, extinction, parameters);

                float3 sample_transmittance = exp(-dt * extinction);

                float3 scattering_no_phase = rayleigh_scattering + mie_scattering;
                float3 scattering_f = (scattering_no_phase - scattering_no_phase * sample_transmittance) / extinction;
                lum_factor += transmittance * scattering_f;

                float3 sun_transmittance = get_val_from_tlut(tlut, new_pos, sun_dir, parameters);

                float3 rayleigh_in_scattering = rayleigh_scattering * rayleigh_phase;
                float mie_in_scattering = mie_scattering * mie_phase_value;
                float3 in_scattering = (rayleigh_in_scattering + mie_in_scattering) * sun_transmittance;

                float3 scattering_integral = (in_scattering - in_scattering * sample_transmittance) / extinction;

                lum += scattering_integral * transmittance;
                transmittance *= sample_transmittance;
            }

            if (ground_dist > 0.0f) {
                float3 hit_pos = normalize(pos + ground_dist * ray_dir) * parameters.ground_radius_mm;
                if (dot(pos, sun_dir) > 0.0f) {
                    lum += transmittance * parameters.ground_albedo.xyz * get_val_from_tlut(tlut, hit_pos, sun_dir, parameters);
                }
            }

            fms += lum_factor * inv_samples;
            lum_total += lum * inv_samples;
        }
    }
}

float3 raymarch_scattering(float3 pos, float3 ray_dir, float3 sun_dir, float tmax, float num_steps, texture2d<float> tlut, texture2d<float> mslut, const device atmosphere_parameters& parameters) {
    float cos_theta = dot(ray_dir, sun_dir);
    float mie_phase_value = get_mie_phase(cos_theta);
    float rayleigh_phase_val = get_rayleigh_phase(-cos_theta);

    float3 lum = float3(0.0f), transmittance = float3(1.0f);
    float t = 0.0f;

    for (float i = 0.0f; i < num_steps; i += 1.0f) {
        float new_t = ((i + 0.3f) / num_steps) * tmax;
        float dt = new_t - t;
        t = new_t;

        float3 new_pos = pos + t * ray_dir;
        float3 rayleigh_scattering = 0.0f, extinction = 0.0f;
        float mie_scattering = 0.0f;
        get_scattering_values(new_pos, rayleigh_scattering, mie_scattering, extinction, parameters);

        float3 sample_transmittance = exp(-dt * extinction);
        float3 sun_transmittance = get_val_from_tlut(tlut, new_pos, sun_dir, parameters);
        float3 psi_ms = get_val_from_mslut(mslut, new_pos, sun_dir, parameters);

        float3 rayleigh_in_scattering = rayleigh_scattering * (rayleigh_phase_val * sun_transmittance + psi_ms);
        float3 mie_in_scattering = mie_scattering * (mie_phase_value * sun_transmittance + psi_ms);
        float3 in_scattering = rayleigh_in_scattering + mie_in_scattering;

        float3 scattering_integral = 0.0f;
        if (length(extinction) < 1e-5f) {
            scattering_integral = in_scattering * dt;
        } else {
            scattering_integral = (in_scattering - in_scattering * sample_transmittance) / extinction;
        }

        lum += scattering_integral * transmittance;
        transmittance *= sample_transmittance;
    }
    return lum;
}

float3 get_val_from_sky_lut(texture2d<float> sky_lut, float3 ray_dir, float3 sun_dir, const device scene_data& scene, const device atmosphere_parameters& parameters) {
    constexpr sampler s(filter::linear, address::clamp_to_edge, coord::normalized);

    float3 up = float3(0.0f, 1.0f, 0.0f);
    float cos_view_up = dot(ray_dir, up);
    float view_zenith_angle = acos(cos_view_up);
    
    float3 view_pos = float3(0.0f, parameters.ground_radius_mm + 0.0002f, 0.0f);
    float height = length(view_pos);

    float inner = max(0.0f, height * height - parameters.ground_radius_mm * parameters.ground_radius_mm);
    float horizon_angle = acos(clamp(sqrt(inner) / height, -1.0f, 1.0f)) - 0.5f * M_PI_F;
    float altitude_angle = 0.5f * M_PI_F - view_zenith_angle + horizon_angle;

    float3 projected_ray_raw = ray_dir - up * dot(ray_dir, up);
    float3 projected_sun = normalize(sun_dir - up * dot(sun_dir, up));

    // Guard against vertical ray_dir where the horizontal projection is zero-length.
    float projected_len = length(projected_ray_raw);
    float cos_theta = (projected_len < 1e-5f) ? 1.0f : dot(projected_ray_raw / projected_len, projected_sun);
    float azimuth_angle = acos(cos_theta);

    if (dot(ray_dir, cross(up, sun_dir)) > 0.0f) {
        azimuth_angle = (2.0f * M_PI_F) - azimuth_angle;
    }

    float u = azimuth_angle / (2.0f * M_PI_F);
    float v = 0.5f + 0.5f * sign(altitude_angle) * sqrt(abs(altitude_angle) * 2.0f / M_PI_F);
    return sky_lut.sample(s, float2(u, v)).rgb;
}

[[kernel]]
void transmittance_lut(texture2d<float, access::write> out_tex [[texture(0)]],
                       const device atmosphere_parameters& parameters [[buffer(0)]],
                       uint2 gid [[thread_position_in_grid]]) {
    uint2 res = uint2(out_tex.get_width(), out_tex.get_height());
    if (gid.x >= res.x || gid.y >= res.y) return;

    float u = (float(gid.x) + 0.5f) / float(res.x);
    float v = (float(gid.y) + 0.5f) / float(res.y);

    float sun_cos_theta = 2.0f * u - 1.0f;
    float sun_theta = safeacos(sun_cos_theta);
    float height = mix(parameters.ground_radius_mm, parameters.atmosphere_radius_mm, v);

    float3 pos = float3(0.0f, height, 0.0f);
    float3 sun_dir = normalize(float3(0.0f, sun_cos_theta, -sin(sun_theta)));

    float3 result = get_sun_transmittance(pos, sun_dir, parameters);
    out_tex.write(float4(result, 1.0f), gid);
}

[[kernel]]
void multiple_scattering_lut(
    texture2d<float> tlut [[texture(0)]],
    texture2d<float, access::write> out_tex [[texture(1)]],
    const device atmosphere_parameters& parameters [[buffer(0)]],
    const device scene_data& data [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint2 res = uint2(out_tex.get_width(), out_tex.get_height());
    if (gid.x >= res.x || gid.y >= res.y) return;

    float u = (float(gid.x) + 0.5f) / float(res.x);
    float v = (float(gid.y) + 0.5f) / float(res.y);

    float sun_cos_theta = 2.0f * u - 1.0f;
    float sun_theta = safeacos(sun_cos_theta);
    float3 prebaked_sun_dir = normalize(float3(0.0f, sun_cos_theta, -sin(sun_theta)));
    float height = mix(parameters.ground_radius_mm, parameters.atmosphere_radius_mm, v);

    float3 pos = float3(0.0f, height, 0.0f);
    float3 lum, f_ms;
    get_mul_scatt_values(pos, prebaked_sun_dir, tlut, lum, f_ms, parameters);

    float3 psi = lum / max(1.0f - f_ms, 0.0001f);
    out_tex.write(float4(psi, 1.0f), gid);
}

[[kernel]]
void sky_view_lut(
    texture2d<float> tlut [[texture(0)]],
    texture2d<float> mslut [[texture(1)]],
    texture2d<float, access::write> out_tex [[texture(2)]],
    const device scene_data& scene [[buffer(0)]],
    const device atmosphere_parameters& parameters [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint2 res = uint2(out_tex.get_width(), out_tex.get_height());
    if (gid.x >= res.x || gid.y >= res.y) return;

    float u = (float(gid.x) + 0.5f) / float(res.x);
    float v = (float(gid.y) + 0.5f) / float(res.y);

    float azimuth_angle = u * 2.0f * M_PI_F;

    // Non-linear altitude mapping (section 5.3)
    float adj_v;
    if (v < 0.5f) {
        float coord = 1.0f - 2.0f * v;
        adj_v = -coord * coord;
    } else {
        float coord = v * 2.0f - 1.0f;
        adj_v = coord * coord;
    }

    // World-space camera position is not in Mm; use a fixed sky-model position just above sea level.
    float3 view_pos = float3(0.0f, parameters.ground_radius_mm + 0.0002f, 0.0f);
    float height = length(view_pos);
    float inner = max(0.0f, height * height - parameters.ground_radius_mm * parameters.ground_radius_mm);
    float horizon_angle = safeacos(clamp(sqrt(inner) / height, -1.0f, 1.0f)) - 0.5f * M_PI_F;
    float altitude_angle = adj_v * 0.5f * M_PI_F - horizon_angle;

    float cos_alt = cos(altitude_angle);
    float3 ray_dir = float3(
        cos_alt * sin(azimuth_angle),
        sin(altitude_angle),
        -cos_alt * cos(azimuth_angle)
    );

    float atmo_dist = ray_intersect_sphere(view_pos, ray_dir, parameters.atmosphere_radius_mm);
    if (atmo_dist < 0.0) {
        out_tex.write(float4(1, 0, 0, 1), gid);
        return;
    }
    
    float ground_dist = ray_intersect_sphere(view_pos, ray_dir, parameters.ground_radius_mm);
    float tmax = (ground_dist > 0.0f) ? ground_dist : atmo_dist;

    float3 actual_sun_dir = normalize(-scene.sun.direction_and_radius.xyz);
    float3 lum = raymarch_scattering(view_pos, ray_dir, actual_sun_dir, tmax, float(num_scattering_steps), tlut, mslut, parameters);
    out_tex.write(float4(lum, 1.0f), gid);
}

// Bakes atmosphere + sun disk into a small (128×128) cubemap.
// Night sky is NOT composited here — see composite_sky_cubemap below.
[[kernel]]
void bake_skybox_cubemap(
    texture2d<float> sky_lut [[texture(0)]],
    texturecube<float, access::write> cubemap [[texture(1)]],
    texture2d<float> tlut [[texture(2)]],
    const device scene_data& scene [[buffer(0)]],
    const device atmosphere_parameters& parameters [[buffer(1)]],
    constant uint& face_size [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]])
{
    if (gid.x >= face_size || gid.y >= face_size || gid.z >= 6) return;

    float2 uv = (float2(gid.xy) + 0.5f) / float(face_size) * 2.0f - 1.0f;
    float3 ray_dir = cube_face_ray_dir(gid.z, uv);

    float3 sun_dir = normalize(-scene.sun.direction_and_radius.xyz);
    float3 lum = get_val_from_sky_lut(sky_lut, ray_dir, sun_dir, scene, parameters) * parameters.sky_intensity;

    // Sun disk
    float cos_sun = dot(ray_dir, sun_dir);
    float min_cos = cos(parameters.sun_disk_size * (M_PI_F / 180.0f));
    float sun_weight;
    if (cos_sun >= min_cos) {
        sun_weight = 1.0f;
    } else {
        float offset = min_cos - cos_sun;
        float bloom = exp(-offset * 50000.0f) * 0.5f + 1.0f / (0.02f + offset * 300.0f) * 0.01f;
        sun_weight = smoothstep(0.002f, 1.0f, bloom);
    }

    cubemap.write(float4(lum, 1.0f), gid.xy, gid.z);
}

// Composites the small atmosphere cubemap with the full-res night sky (stars + milky way).
// All inputs are pre-baked cubemaps so this pass is entirely cheap texture fetches + lerp.
//   texture(0) = atmosphere cubemap (128×128, from bake_skybox_cubemap)
//   texture(1) = output composite cubemap (2048×2048, write)
//   texture(2) = star cubemap (2048×2048, from NightSkyPass)
//   texture(3) = milky way cubemap (1024×1024, baked once at startup)
[[kernel]]
void composite_sky_cubemap(
    texturecube<float, access::sample> atmo_cube  [[texture(0)]],
    texturecube<float, access::write>  output     [[texture(1)]],
    texturecube<float, access::sample> star_cube  [[texture(2)]],
    texturecube<float, access::sample> mw_cube    [[texture(3)]],
    const device atmosphere_parameters& parameters [[buffer(0)]],
    constant uint& face_size [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]])
{
    if (gid.x >= face_size || gid.y >= face_size || gid.z >= 6) return;

    float2 uv = (float2(gid.xy) + 0.5f) / float(face_size) * 2.0f - 1.0f;
    float3 ray_dir = cube_face_ray_dir(gid.z, uv);

    constexpr sampler smp(filter::linear, address::clamp_to_edge);

    float3 atmo  = atmo_cube.sample(smp, ray_dir).rgb;
    float3 stars = star_cube.sample(smp, ray_dir).rgb;
    // mw_cube stores raw luminance (baked without exposure); apply at composite time for runtime control
    float3 mw    = mw_cube.sample(smp, ray_dir).rgb * parameters.milkyway_exposure;
    float3 night = stars + mw;

    float3 result = mix(night, atmo, saturate(parameters.night_blend_factor));
    output.write(float4(result, 1.0f), gid.xy, gid.z);
}

[[kernel]]
void sky_draw(
    texture2d<float> depth [[texture(0)]],
    texturecube<float, access::sample> sky_cubemap [[texture(1)]],
    texture2d<float, access::write> hdr [[texture(2)]],
    const device scene_data& scene [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= hdr.get_width() || gid.y >= hdr.get_height()) return;

    float d = depth.read(gid).r;
    if (d < 1.0f) return;

    float2 uv = (float2(gid) + 0.5f) / float2(hdr.get_width(), hdr.get_height());
    float2 ndc = uv * 2.0f - 1.0f;
    ndc.y = -ndc.y;

    float4 clip = float4(ndc, 1.0f, 1.0f);
    float4 world_h = scene.camera.inverse_view_projection * clip;
    float3 world_pos = world_h.xyz / world_h.w;
    float3 ray_dir = normalize(world_pos - scene.camera.position_and_near.xyz);

    constexpr sampler smp(filter::linear, address::clamp_to_edge);
    float3 sky = sky_cubemap.sample(smp, ray_dir).rgb;

    hdr.write(float4(sky, 1.0f), gid);
}
