//
//  Triangle.metal
//  Neural Graphics
//
//  Created by Amélie Heinrich on 25/02/2026.
//

#include <metal_stdlib>
using namespace metal;

struct VSOut
{
    float4 Position [[position]];
    float3 Color;
};

[[vertex]]
VSOut triangle_vs(uint id [[vertex_id]]) {
    float3 positions[] = {
        float3(-0.5f, -0.5f, 0.0f),
        float3(0.5f, -0.5f, 0.0f),
        float3(0.0f, 0.5f, 0.0f)
    };
    
    float3 colors[] = {
        float3(1.0f, 0.0f, 0.0f),
        float3(0.0f, 1.0f, 0.0f),
        float3(0.0f, 0.0f, 1.0f)
    };
    
    VSOut out;
    out.Position = float4(positions[id], 1.0f);
    out.Color = colors[id];
    return out;
}

[[fragment]]
float4 triangle_fs(VSOut in [[stage_in]]) {
    return float4(in.Color, 1.0f);
}
