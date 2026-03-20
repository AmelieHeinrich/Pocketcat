//
//  Tonemap.metal
//  Pocketcat
//
//  Created by Amélie Heinrich on 02/03/2026.
//

#include <metal_stdlib>
#include <simd/simd.h>
using namespace simd;
using namespace metal;

struct Parameters {
    float Gamma;
};

// I'm using ACES Narcowicz here. TODO: Neural tonemap? >.>
float3 Tonemap(float3 x) {
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return saturate((x*(a*x+b))/(x*(c*x+d)+e));
}

struct TonemapVSOut {
    float4 position [[position]];
    float2 uv;
};

[[vertex]]
TonemapVSOut tonemap_vs(uint vid [[vertex_id]]) {
    float2 positions[3] = {
        float2(-1.0,  1.0),
        float2( 3.0,  1.0),
        float2(-1.0, -3.0)
    };
    float2 uvs[3] = {
        float2(0.0, 0.0),
        float2(2.0, 0.0),
        float2(0.0, 2.0)
    };
    TonemapVSOut out;
    out.position = float4(positions[vid], 0.0, 1.0);
    out.uv = uvs[vid];
    return out;
}

[[fragment]]
float4 tonemap_fs(
    TonemapVSOut in [[stage_in]],
    texture2d<float> input [[texture(0)]],
    constant Parameters& params [[buffer(0)]]
) {
    constexpr sampler s(filter::nearest, address::clamp_to_edge);
    float3 color = input.sample(s, in.uv).xyz;
    float3 mappedColor = Tonemap(color);
    mappedColor = pow(mappedColor, 1.0 / params.Gamma);
    return float4(color, 1.0);
}
