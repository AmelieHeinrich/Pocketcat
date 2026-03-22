//
//  PrimaryRayTest.metal
//  Pocketcat
//
//  Created by Amélie Heinrich on 20/03/2026.
//

#include "Common/Bindless.h"
#include "Common/Math.h"
#include "Common/PBR.h"

// ─── PCG hash RNG ─────────────────────────────────────────────────────────────

struct RNG {
    uint state;

    uint next() {
        state = state * 747796405u + 2891336453u;
        uint w = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (w >> 22u) ^ w;
    }

    float next_f() {
        return float(next()) / float(0xFFFFFFFFu);
    }
};

RNG make_rng(uint2 pid, uint frame) {
    RNG rng;
    uint h = pid.x * 1973u + pid.y * 9277u + frame * 26699u;
    rng.state = h ^ (h >> 16u);
    return rng;
}

// ─── Cosine-weighted hemisphere sample ────────────────────────────────────────

float3 sample_cosine_hemisphere(float3 normal, float r1, float r2) {
    float sin_theta = sqrt(r1);
    float cos_theta = sqrt(1.0 - r1);
    float phi = 2.0 * M_PI_F * r2;

    float3 local = float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);

    float3 up = abs(normal.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 t = normalize(cross(up, normal));
    float3 b = cross(normal, t);

    return normalize(t * local.x + b * local.y + normal * local.z);
}

// ─── Visibility (any-hit shadow ray) ─────────────────────────────────────────

float visibility(float3 origin, float3 dir, float max_dist,
                 const device scene_data& scene,
                 intersection_function_table<triangle_data, instancing> ift)
{
    ray r;
    r.origin = origin;
    r.direction = dir;
    r.min_distance = 0.001;
    r.max_distance = max_dist;

    intersector<triangle_data, instancing> si;
    si.assume_geometry_type(geometry_type::triangle);
    si.accept_any_intersection(true);

    return (si.intersect(r, scene.tlas, 0xFF, ift).type == intersection_type::none) ? 1.0 : 0.0;
}

// ─── Surface data ─────────────────────────────────────────────────────────────

struct SurfaceHit {
    float3 pos;
    float3 n;
    float3 albedo;
    float  roughness;
    float  metallic;
    float  ao;
    float3 emissive;
};

float3 eval_brdf(SurfaceHit hit, float3 v, float3 l) {
    float3 h = normalize(v + l);
    float n_dot_l = saturate(dot(hit.n, l));
    float n_dot_v = saturate(dot(hit.n, v));
    float n_dot_h = saturate(dot(hit.n, h));
    float v_dot_h = saturate(dot(v, h));

    float3 f0 = mix(float3(0.04), hit.albedo, hit.metallic);
    float3 f = f_schlick(v_dot_h, f0);
    float  d = d_ggx(n_dot_h, hit.roughness);
    float  g = g_smith(n_dot_v, n_dot_l, hit.roughness);

    float3 specular = (d * g * f) / max(4.0 * n_dot_v * n_dot_l, 0.0001);
    float3 kd = (1.0 - f) * (1.0 - hit.metallic);
    float3 diffuse = kd * hit.albedo / M_PI_F;

    return (diffuse + specular) * n_dot_l;
}

// ─── Secondary-hit surface fetch ──────────────────────────────────────────────

SurfaceHit fetch_secondary_hit(const device scene_data& scene,
                                uint instance_id, uint primitive_id, float2 bary,
                                float3 hit_pos, float3 ray_dir)
{
    constexpr sampler s(mag_filter::linear, min_filter::linear,
                        address::repeat, mip_filter::linear);

    instance  inst = scene.instances[instance_id];
    material  mat  = scene.materials[inst.material_index];
    triangle  tri  = fetch_triangle(scene, instance_id, primitive_id);

    float2 uv    = interpolate2D(bary, tri.v0.uv,      tri.v1.uv,      tri.v2.uv);
    float3 n_geo = interpolate2D(bary, tri.v0.normal,  tri.v1.normal,  tri.v2.normal);
    float4 t_geo = interpolate2D(bary, tri.v0.tangent, tri.v1.tangent, tri.v2.tangent);

    if (dot(n_geo, -ray_dir) < 0.0) n_geo = -n_geo;

    float4 albedo_sample = mat.has_albedo() ? mat.albedo.sample(s, uv) : float4(0.8, 0.8, 0.8, 1.0);
    float3 orm           = mat.has_orm()    ? mat.orm.sample(s, uv).rgb : float3(1, 0.5, 0);

    float3 n = n_geo;
    if (mat.has_normal()) {
        float3 t = normalize(t_geo.xyz - dot(t_geo.xyz, n_geo) * n_geo);
        float3 b = cross(n_geo, t) * t_geo.w;
        float3x3 tbn = float3x3(t, b, n_geo);
        float3 nmap = mat.normal.sample(s, uv).xyz * 2.0 - 1.0;
        n = normalize(tbn * nmap);
    }

    SurfaceHit h;
    h.pos       = hit_pos;
    h.n         = n;
    h.albedo    = albedo_sample.rgb;
    h.ao        = orm.r;
    h.roughness = clamp(orm.g, 0.04, 1.0);
    h.metallic  = orm.b;
    h.emissive  = mat.has_emissive() ? mat.emissive.sample(s, uv).rgb : float3(0);
    return h;
}

// ─── Pathtracer ───────────────────────────────────────────────────────────────

[[kernel]]
void pathtracer(const device scene_data& scene          [[buffer(0)]],
                intersection_function_table<triangle_data, instancing> ift [[buffer(1)]],
                constant uint& frame_index               [[buffer(2)]],
                texture2d<float> depth_texture           [[texture(0)]],
                texture2d<float> albedo_texture          [[texture(1)]],
                texture2d<float> normal_texture          [[texture(2)]],
                texture2d<float> orm_texture             [[texture(3)]],
                texture2d<float> emissive_texture        [[texture(4)]],
                texture2d<float, access::write> output   [[texture(5)]],
                uint2 pid [[thread_position_in_grid]])
{
    const float3 light_dir = normalize(float3(0.3, -1.0, 0.2));
    const float3 light_color = float3(1.0, 0.95, 0.85) * 3.0;

    uint width = output.get_width();
    uint height = output.get_height();
    if (pid.x >= width || pid.y >= height) return;

    float2 dimensions = float2(width, height);
    float2 pixel_center = float2(pid) + 0.5;
    float2 uv = pixel_center / dimensions;
    float2 ndc = uv * 2.0 - 1.0;
    ndc.y = -ndc.y;

    // Sky

    float depth = depth_texture.read(pid).r;
    if (depth >= 1.0) {
        float4 far4    = scene.camera.inverse_view_projection * float4(ndc, 1.0, 1.0);
        float3 far_w   = far4.xyz / far4.w;
        float3 cam_pos = scene.camera.position_and_near.xyz;
        float3 sky_dir = normalize(far_w - cam_pos);
        float t = saturate(0.5 * (sky_dir.y + 1.0));
        float3 sky = mix(float3(1.0), float3(0.5, 0.7, 1.0), t);
        output.write(float4(sky, 1.0), pid);
        return;
    }

    // Reconstruct

    float4 clip4     = float4(ndc, depth, 1.0);
    float4 world4    = scene.camera.inverse_view_projection * clip4;
    float3 world_pos = world4.xyz / world4.w;

    SurfaceHit hit0;
    hit0.pos      = world_pos;
    hit0.n        = normalize(normal_texture.read(pid).xyz);
    hit0.albedo   = albedo_texture.read(pid).rgb;
    float2 rm     = orm_texture.read(pid).rg;
    hit0.roughness = max(rm.r, 0.04);
    hit0.metallic  = rm.g;
    hit0.ao        = 1.0;
    hit0.emissive  = emissive_texture.read(pid).rgb;

    float3 cam_pos = scene.camera.position_and_near.xyz;
    float3 v0      = normalize(cam_pos - world_pos);

    RNG rng = make_rng(pid, frame_index);

    intersector<triangle_data, instancing> inter;
    inter.assume_geometry_type(geometry_type::triangle);

    // NEE
    float3 L_direct = float3(0);
    {
        float vis = visibility(hit0.pos + hit0.n * 0.001, -light_dir, 10000.0, scene, ift);
        L_direct = eval_brdf(hit0, v0, -light_dir) * light_color * vis;
    }

    // Indirect
    float3 L_indirect = float3(0);
    {
        float3 wi  = sample_cosine_hemisphere(hit0.n, rng.next_f(), rng.next_f());
        float  pdf = saturate(dot(hit0.n, wi)) / M_PI_F;

        if (pdf > 0.0) {
            ray bounce;
            bounce.origin       = hit0.pos + hit0.n * 0.001;
            bounce.direction    = wi;
            bounce.min_distance = 0.001;
            bounce.max_distance = 10000;

            auto bounce_result = inter.intersect(bounce, scene.tlas, 0xFF, ift);

            float3 Li = float3(0);
            if (bounce_result.type == intersection_type::none) {
                float t = saturate(0.5 * (wi.y + 1.0));
                Li = mix(float3(1.0), float3(0.5, 0.7, 1.0), t);
            } else {
                float3 hit_pos1 = hit0.pos + wi * bounce_result.distance;
                SurfaceHit hit1 = fetch_secondary_hit(
                    scene,
                    bounce_result.instance_id,
                    bounce_result.primitive_id,
                    bounce_result.triangle_barycentric_coord,
                    hit_pos1, wi);

                float vis1 = visibility(hit1.pos + hit1.n * 0.001, -light_dir, 10000.0, scene, ift);
                Li = hit1.emissive + eval_brdf(hit1, -wi, -light_dir) * light_color * vis1;
            }

            float3 brdf_val = eval_brdf(hit0, v0, wi);
            L_indirect = brdf_val * Li / pdf;
        }
    }

    float3 color = hit0.emissive + L_direct + L_indirect;
    float lum = dot(color, float3(0.2126, 0.7152, 0.0722));
    if (lum > 10.0) color *= 10.0 / lum;

    output.write(float4(color, 1.0), pid);
}
