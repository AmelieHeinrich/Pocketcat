//
//  Triangle.metal
//  Neural Graphics
//
//  Created by Amélie Heinrich on 25/02/2026.
//

#include <metal_stdlib>
using namespace metal;

struct VSOut {
    float4 Position [[position]];
    float2 UV;
};

[[vertex]]
VSOut triangle_vs(uint id [[vertex_id]]) {
    float3 positions[] = {
        float3( 0.5f,  0.5f, 0.0f),
        float3( 0.5f, -0.5f, 0.0f),
        float3(-0.5f, -0.5f, 0.0f),
        float3(-0.5f,  0.5f, 0.0f)
    };
    
    float2 uvs[] = {
        float2(1.0f, 1.0f),
        float2(1.0f, 0.0f),
        float2(0.0f, 0.0f),
        float2(0.0f, 1.0f)
    };
    
    VSOut out;
    out.Position = float4(positions[id], 1.0f);
    out.UV = uvs[id];
    return out;
}

[[fragment]]
float4 triangle_fs(VSOut in [[stage_in]],
                   texture2d<float> albedo [[texture(0)]]) {
    constexpr sampler textureSampler(
        mag_filter::linear,
        min_filter::linear,
        mip_filter::linear,
        address::repeat,
        lod_clamp(0.0f, MAXFLOAT)
    );
    
    return albedo.sample(textureSampler, in.UV);
}
