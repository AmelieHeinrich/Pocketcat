//
//  TextureViz.metal
//  Pocketcat
//
//  Created by Amélie Heinrich on 01/04/2026.
//

#include <metal_stdlib>
using namespace metal;

#include "Common/Math.h"

struct texviz_vs_out {
    float4 position [[position]];
    float2 uv;
};

// rect = (ndc_x_left, ndc_y_top, ndc_width, ndc_height_positive)
// 6 vertices forming 2 triangles (triangle list)
[[vertex]]
texviz_vs_out texviz_vs(uint vid             [[vertex_id]],
                        constant float4& rect [[buffer(0)]]) {
    float2 corners[6] = { {0,0},{1,0},{0,1},  {1,0},{1,1},{0,1} };
    float2 uvs[6]     = { {0,0},{1,0},{0,1},  {1,0},{1,1},{0,1} };
    float2 c = corners[vid];
    // NDC y goes up; c.y==0 → top of thumbnail, c.y==1 → bottom
    float2 ndc = float2(rect.x + c.x * rect.z,
                        rect.y - c.y * rect.w);
    texviz_vs_out o;
    o.position = float4(ndc, 0.0, 1.0);
    o.uv = uvs[vid];
    return o;
}

[[fragment]]
float4 texviz_passthrough_fs(texviz_vs_out in [[stage_in]],
                             texture2d<float> input [[texture(0)]]) {
    constexpr sampler s(filter::nearest, address::clamp_to_edge);
    return float4(input.sample(s, in.uv).rgb, 1.0);
}

// -------------------------------------------------------------------------
// Visibility buffer — integer texture (rg32Uint)
// R = (meshlet_index << 8) | (prim_id & 0xFF)   [mesh shader]
//   = 0x80000000 | prim_id                       [vertex shader]
// G = instance_id
// -------------------------------------------------------------------------

[[fragment]]
float4 texviz_visibility_instance_id_fs(texviz_vs_out in [[stage_in]],
                                        texture2d<uint> input [[texture(0)]]) {
    uint2 coord = uint2(in.uv * float2(input.get_width(), input.get_height()));
    uint instance_id = input.read(coord).g;
    return float4(hash_color(instance_id), 1.0);
}

[[fragment]]
float4 texviz_visibility_meshlet_id_fs(texviz_vs_out in [[stage_in]],
                                       texture2d<uint> input [[texture(0)]]) {
    uint2 coord = uint2(in.uv * float2(input.get_width(), input.get_height()));
    uint r = input.read(coord).r;
    // top bit set means vertex-shader path — no meshlet info
    if (r & 0x80000000u) return float4(0.4, 0.4, 0.4, 1.0);
    uint meshlet_id = r >> 8u;
    return float4(hash_color(meshlet_id), 1.0);
}

[[fragment]]
float4 texviz_visibility_primitive_id_fs(texviz_vs_out in [[stage_in]],
                                         texture2d<uint> input [[texture(0)]]) {
    uint2 coord = uint2(in.uv * float2(input.get_width(), input.get_height()));
    uint r = input.read(coord).r;
    uint prim_id = (r & 0x80000000u) ? (r & 0x7FFFFFFFu) : (r & 0xFFu);
    return float4(hash_color(prim_id), 1.0);
}

// -------------------------------------------------------------------------
// GBuffer helpers
// -------------------------------------------------------------------------

[[fragment]]
float4 texviz_depth_fs(texviz_vs_out in [[stage_in]],
                       texture2d<float> input [[texture(0)]]) {
    constexpr sampler s(filter::nearest, address::clamp_to_edge);
    float d = input.sample(s, in.uv).r;
    d = pow(d, 5.0);
    return float4(d, d, d, 1.0);
}

[[fragment]]
float4 texviz_motion_vectors_fs(texviz_vs_out in [[stage_in]],
                                texture2d<float> input [[texture(0)]]) {
    constexpr sampler s(filter::nearest, address::clamp_to_edge);
    float2 mv = abs(input.sample(s, in.uv).rg) * 10.0;
    return float4(mv.x, mv.y, 0.0, 1.0);
}

[[fragment]]
float4 texviz_gbuffer_normal_fs(texviz_vs_out in [[stage_in]],
                                texture2d<float> input [[texture(0)]]) {
    constexpr sampler s(filter::nearest, address::clamp_to_edge);
    float3 n = input.sample(s, in.uv).rgb;
    return float4(n * 0.5 + 0.5, 1.0);
}

[[fragment]]
float4 texviz_gbuffer_roughness_fs(texviz_vs_out in [[stage_in]],
                                   texture2d<float> input [[texture(0)]]) {
    constexpr sampler s(filter::nearest, address::clamp_to_edge);
    float r = input.sample(s, in.uv).r;
    return float4(r, r, r, 1.0);
}

[[fragment]]
float4 texviz_gbuffer_metallic_fs(texviz_vs_out in [[stage_in]],
                                  texture2d<float> input [[texture(0)]]) {
    constexpr sampler s(filter::nearest, address::clamp_to_edge);
    float m = input.sample(s, in.uv).g;
    return float4(m, m, m, 1.0);
}

// -------------------------------------------------------------------------
// Generic single-channel (R → grayscale).  Works for r8Unorm and rg16Float.
// -------------------------------------------------------------------------

[[fragment]]
float4 texviz_single_channel_fs(texviz_vs_out in [[stage_in]],
                                texture2d<float> input [[texture(0)]]) {
    constexpr sampler s(filter::nearest, address::clamp_to_edge);
    float v = input.sample(s, in.uv).r;
    return float4(v, v, v, 1.0);
}

// -------------------------------------------------------------------------
// HDR float textures — Reinhard tone map
// -------------------------------------------------------------------------

[[fragment]]
float4 texviz_hdr_fs(texviz_vs_out in [[stage_in]],
                     texture2d<float> input [[texture(0)]]) {
    constexpr sampler s(filter::nearest, address::clamp_to_edge);
    float3 c = input.sample(s, in.uv).rgb;
    c = c / (1.0 + c);
    return float4(c, 1.0);
}

// -------------------------------------------------------------------------
// RT Shadows denoiser history
// -------------------------------------------------------------------------

[[fragment]]
float4 texviz_shadow_history_length_fs(texviz_vs_out in [[stage_in]],
                                       texture2d<float> input [[texture(0)]]) {
    constexpr sampler s(filter::nearest, address::clamp_to_edge);
    float h = saturate(input.sample(s, in.uv).r / 32.0);
    return float4(h, h, h, 1.0);
}

[[fragment]]
float4 texviz_shadow_history_acceptance_fs(texviz_vs_out in [[stage_in]],
                                           texture2d<float> input [[texture(0)]]) {
    constexpr sampler s(filter::nearest, address::clamp_to_edge);
    float h = input.sample(s, in.uv).r;
    // red = rejected/reset (history_length <= 1), blue = accepted
    return h <= 1.0 ? float4(1.0, 0.0, 0.0, 1.0) : float4(0.0, 0.0, 1.0, 1.0);
}
