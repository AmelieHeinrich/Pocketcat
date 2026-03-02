//
//  Model.metal
//  Neural Graphics
//
//  Created by Amélie Heinrich on 02/03/2026.
//

#include <metal_stdlib>
using namespace metal;

// MeshVertex on the CPU is tightly packed (float[3] = 12 bytes each).
// Metal's float3 has implicit 4-byte padding making sizeof = 16, which breaks
// the stride. packed_float3 matches the C++ layout: 12 bytes, no padding.
struct VSIn {
    packed_float3 Position;
    packed_float3 Normal;
    float2        UV;
    float4        Tangent;
};

struct VSOut {
    float4 Position [[position]];
    float2 UV;
};

struct ModelData {
    float4x4 Camera;
    uint VertexOffset;
};

[[vertex]]
VSOut triangle_vs(uint id [[vertex_id]],
                  const device ModelData& modelData [[buffer(0)]],
                  const device VSIn* vertices [[buffer(1)]]) {
    uint index = modelData.VertexOffset + id;

    VSOut out;
    out.Position = modelData.Camera * float4(vertices[index].Position, 1.0f);
    out.UV = vertices[index].UV;
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

    float4 color = albedo.sample(textureSampler, in.UV);
    if (color.a < 0.25)
        discard_fragment();
    
    return color;
}
