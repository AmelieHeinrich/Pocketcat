/**
 * NightSky.metal
 * Pocketcat
 *
 * Three shaders:
 *   1. nightsky_bake_milkyway  – compute, runs once: equirectangular → cubemap with galactic transform
 *   2. nightsky_stars_vs       – vertex, draws star quads into a layered cubemap render target
 *   3. nightsky_stars_fs       – fragment, applies B-V colour + sprite alpha + brightness
 */

#include <metal_stdlib>
using namespace metal;

// -----------------------------------------------------------------------
// Shared star type (matches StarEntry in StarCatalogCompressor.h)
// -----------------------------------------------------------------------
struct Star {
    float ra;           // radians
    float dec;          // radians
    float mag;          // apparent magnitude
    float bv;           // B-V colour index
};

// -----------------------------------------------------------------------
// Cubemap face constants
// -----------------------------------------------------------------------
#define FACE_PX 0   // +X
#define FACE_NX 1   // -X
#define FACE_PY 2   // +Y
#define FACE_NY 3   // -Y
#define FACE_PZ 4   // +Z
#define FACE_NZ 5   // -Z

inline float3 ra_dec_to_dir(float ra, float dec)
{
    float cos_dec = cos(dec);
    return float3(cos(ra) * cos_dec, sin(dec), sin(ra) * cos_dec);
}

inline uint select_face(float3 d)
{
    float3 a = abs(d);
    if (a.x >= a.y && a.x >= a.z) return d.x > 0.0 ? FACE_PX : FACE_NX;
    if (a.y >= a.x && a.y >= a.z) return d.y > 0.0 ? FACE_PY : FACE_NY;
    return d.z > 0.0 ? FACE_PZ : FACE_NZ;
}

inline float2 dir_to_face_uv(float3 d, uint face)
{
    float u, v;
    switch (face) {
        case FACE_PX: u = -d.z / abs(d.x); v = -d.y / abs(d.x); break;
        case FACE_NX: u =  d.z / abs(d.x); v = -d.y / abs(d.x); break;
        case FACE_PY: u =  d.x / abs(d.y); v =  d.z / abs(d.y); break;
        case FACE_NY: u =  d.x / abs(d.y); v = -d.z / abs(d.y); break;
        case FACE_PZ: u =  d.x / abs(d.z); v = -d.y / abs(d.z); break;
        default:      u = -d.x / abs(d.z); v = -d.y / abs(d.z); break;
    }
    return float2(0.5 * (u + 1.0), 0.5 * (v + 1.0));
}

// -----------------------------------------------------------------------
// 1. Milky Way bake: equirectangular texture → cubemap (one-shot compute)
//    buffer(0)  = NightSkyBakeParams { milkyway_exposure }
//    texture(0) = milkyway equirectangular (2D, sRGB)
//    texture(1) = output cubemap (write)
// -----------------------------------------------------------------------
struct NightSkyBakeParams {
    float milkyway_exposure;
};

kernel void nightsky_bake_milkyway(
    const device NightSkyBakeParams& params [[buffer(0)]],
    texture2d<float, access::sample>         milkyway [[texture(0)]],
    texturecube<float, access::write>        output   [[texture(1)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint face = gid.z;
    uint size = output.get_width();
    if (gid.x >= size || gid.y >= size) return;

    // Reconstruct world direction for this texel
    float2 uv = (float2(gid.xy) + 0.5) / float(size);
    float2 ndc = uv * 2.0 - 1.0;
    float3 dir;
    switch (face) {
        case FACE_PX: dir = normalize(float3( 1.0, -ndc.y, -ndc.x)); break;
        case FACE_NX: dir = normalize(float3(-1.0, -ndc.y,  ndc.x)); break;
        case FACE_PY: dir = normalize(float3( ndc.x,  1.0,  ndc.y)); break;
        case FACE_NY: dir = normalize(float3( ndc.x, -1.0, -ndc.y)); break;
        case FACE_PZ: dir = normalize(float3( ndc.x, -ndc.y,  1.0)); break;
        default:      dir = normalize(float3(-ndc.x, -ndc.y, -1.0)); break;
    }

    // IAU J2000 equatorial → galactic rotation
    float3x3 eq_to_gal = float3x3(
        float3(-0.0548755604,  0.4941094279, -0.8676661490),
        float3(-0.8734370902, -0.4448296300, -0.1980763734),
        float3(-0.4838350155,  0.7469822445,  0.4559837762)
    );
    float3 gal = eq_to_gal * dir;

    float b = asin(clamp(gal.z, -1.0, 1.0));
    float l = atan2(gal.y, gal.x);
    float2 mw_uv = float2(fract(l / (2.0 * M_PI_F) + 0.5), b / M_PI_F + 0.5);

    constexpr sampler smp(filter::linear, address::repeat);
    float3 mw = milkyway.sample(smp, mw_uv).rgb * params.milkyway_exposure;

    output.write(float4(mw, 1.0), gid.xy, face);
}

// -----------------------------------------------------------------------
// 2 & 3. Star rendering: instanced quads into a layered cubemap RT
//    buffer(0)  = Star[]
//    buffer(1)  = NightStarParams { star_brightness, cubemap_size }
//    texture(0) = airy disk sprite (fragment only)
// -----------------------------------------------------------------------
struct NightStarParams {
    float star_brightness;
    uint  cubemap_size;
};

struct StarVsOut {
    float4 position [[position]];
    float2 uv;
    uint   face [[render_target_array_index]];
    uint   star_index;
};

// Two CCW triangles forming a billboard quad
constant float2 kQuadCorners[6] = {
    float2(-1.0,  1.0), float2( 1.0,  1.0), float2(-1.0, -1.0),
    float2( 1.0,  1.0), float2( 1.0, -1.0), float2(-1.0, -1.0)
};
constant float2 kQuadUVs[6] = {
    float2(0.0, 0.0), float2(1.0, 0.0), float2(0.0, 1.0),
    float2(1.0, 0.0), float2(1.0, 1.0), float2(0.0, 1.0)
};

[[vertex]]
StarVsOut nightsky_stars_vs(
    const device Star*           stars  [[buffer(0)]],
    const device NightStarParams& params [[buffer(1)]],
    uint vid [[vertex_id]],
    uint iid [[instance_id]])
{
    StarVsOut out;
    Star star = stars[iid];

    // Clip out-of-range stars (e.g. the Sun entry with mag < -5)
    if (star.mag < -5.0) {
        out.position = float4(10.0, 10.0, 10.0, 1.0);
        out.uv = float2(0.0);
        out.face = 0;
        out.star_index = iid;
        return out;
    }

    float3 dir    = ra_dec_to_dir(star.ra, star.dec);
    uint   face   = select_face(dir);
    float2 center = dir_to_face_uv(dir, face) * 2.0 - 1.0;  // NDC

    float half_px  = max(1.5, 5.0 - star.mag);
    float half_ndc = half_px / float(params.cubemap_size);

    out.position    = float4(center + kQuadCorners[vid] * half_ndc, 0.0, 1.0);
    out.uv          = kQuadUVs[vid];
    out.face        = face;
    out.star_index  = iid;
    return out;
}

inline float3 bv_to_rgb(float bv)
{
    bv = clamp(bv, -0.4, 2.0);
    float T = 4600.0 * (1.0 / (0.92 * bv + 1.7) + 1.0 / (0.92 * bv + 0.62));
    float t = T / 100.0;

    float r = (t <= 66.0) ? 1.0 : 1.292936186062745 * pow(t - 60.0, -0.1332047592);
    float g = (t <= 66.0) ? 0.3900815787690196 * log(t) - 0.6318414437886275
                           : 1.129890860895294  * pow(t - 60.0, -0.0755148492);
    float b;
    if      (t >= 66.0) b = 1.0;
    else if (t <= 19.0) b = 0.0;
    else                b = 0.5432067891101961 * log(t - 10.0) - 1.19625408914;

    return max(float3(r, g, b), float3(0.0));
}

[[fragment]]
float4 nightsky_stars_fs(
    StarVsOut                     in     [[stage_in]],
    const device Star*            stars  [[buffer(0)]],
    const device NightStarParams& params [[buffer(1)]],
    texture2d<float>              sprite [[texture(0)]])
{
    constexpr sampler smp(filter::linear, address::clamp_to_zero);
    float alpha = sprite.sample(smp, in.uv).r;

    Star  star       = stars[in.star_index];
    float mag        = clamp(star.mag, -1.5, 7.5);
    float brightness = pow(10.0, -0.4 * mag) * params.star_brightness;
    float3 col       = bv_to_rgb(star.bv);

    return float4(col * brightness, alpha);
}
