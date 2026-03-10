//
//  Model.metal
//  Neural Graphics
//
//  Created by Amélie Heinrich on 02/03/2026.
//

#include "../Common/Bindless.h"

#include <metal_stdlib>
using namespace metal;

struct VSOut {
    float4 Position [[position]];
    float2 UV;
    float3 Normal;
    float3 WorldPos;
    float4 Tangent;
    int MaterialIndex [[flat]];
};

[[vertex]]
VSOut forward_vs(uint vid [[vertex_id]],
                 const device SceneBuffer& scene [[buffer(0)]],
                 const device uint& instanceID [[buffer(1)]],
                 uint instanceIndex [[base_instance]]) {
    if (instanceIndex >= scene.InstanceCount) {
        VSOut out;
        out.MaterialIndex = -1;
        return out;
    }
    
    SceneInstance inst = scene.Instances[instanceIndex];
    SceneEntity entity = scene.Entities[inst.EntityIndex];

    MeshVertex v = inst.VertexBuffer[vid];

    float4 worldPos = entity.Transform * float4(v.Position, 1.0f);

    VSOut out;
    out.Position      = scene.Camera.ViewProjection * worldPos;
    out.UV            = v.UV;
    out.Normal        = normalize((entity.Transform * float4(v.Normal, 0.0f)).xyz);
    out.WorldPos      = worldPos.xyz;
    out.Tangent       = v.Tangent;
    out.MaterialIndex = inst.MaterialIndex;
    return out;
}

[[fragment]]
float4 forward_vsfs(VSOut in [[stage_in]],
                    const device SceneBuffer& scene [[buffer(0)]]) {
    if (in.MaterialIndex == -1) return float4(0.0f);
    
    constexpr sampler textureSampler(
        mag_filter::linear,
        min_filter::linear,
        mip_filter::linear,
        address::repeat,
        lod_clamp(0.0f, MAXFLOAT)
    );

    SceneMaterial mat = scene.Materials[in.MaterialIndex];

    // Albedo
    float4 color = float4(1.0f);
    if (mat.hasAlbedo()) {
        color = mat.Albedo.sample(textureSampler, in.UV);
    }

    if (color.a < 0.25f)
        discard_fragment();

    return color;
}
