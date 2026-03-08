//
//  ModelMS.metal
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 03/03/2026.
//

#include "Common/Bindless.h"

#include <metal_stdlib>
using namespace metal;

#define AS_GROUP_SIZE 32

struct ForwardPushConstants {
    uint InstanceIndex;
    uint LOD;
};

struct Payload {
    uint MeshletIndices[AS_GROUP_SIZE];
};

struct MSOut {
    float4 Position [[position]];
    float2 UV;
    float3 Normal;
    float3 WorldPos;
    float4 Tangent;
    uint   InstanceIndex [[flat]];
};

using MeshOutput = metal::mesh<MSOut, void, 64, 128, topology::triangle>;

[[object]]
void forward_os(uint                 gtid       [[thread_position_in_threadgroup]],
                uint                 dtid       [[thread_position_in_grid]],
                object_data Payload& outPayload [[payload]],
                mesh_grid_properties outGrid) {
    outPayload.MeshletIndices[gtid] = dtid;
    outGrid.set_threadgroups_per_grid(uint3(AS_GROUP_SIZE, 1, 1));
}

[[mesh]]
void forward_ms(const device SceneBuffer& scene [[buffer(0)]],
                const device ForwardPushConstants& push [[buffer(1)]],
                object_data const Payload& payload [[payload]],
                uint gtid [[thread_position_in_threadgroup]],
                uint gid [[threadgroup_position_in_grid]],
                MeshOutput outMesh) {
    SceneInstance inst = scene.Instances[push.InstanceIndex];
    SceneEntity entity = scene.Entities[inst.EntityIndex];
    SceneInstanceLOD lod = inst.LODs[push.LOD];

    uint meshletIndex = payload.MeshletIndices[gid];
    if (meshletIndex >= lod.MeshletCount) {
        outMesh.set_primitive_count(0);
        return;
    }

    MeshMeshlet m = lod.Meshlets[meshletIndex];
    outMesh.set_primitive_count(m.TriangleCount);

    if (gtid < m.TriangleCount) {
        uint triBase = m.TriangleOffset + gtid * 3;
        uint vIdx0 = lod.MeshletTriangles[triBase + 0];
        uint vIdx1 = lod.MeshletTriangles[triBase + 1];
        uint vIdx2 = lod.MeshletTriangles[triBase + 2];

        uint triIdx = 3 * gtid;
        outMesh.set_index(triIdx + 0, vIdx0);
        outMesh.set_index(triIdx + 1, vIdx1);
        outMesh.set_index(triIdx + 2, vIdx2);
    }

    if (gtid < m.VertexCount) {
        uint vertexIndex = m.VertexOffset + gtid;
        MeshVertex v = lod.MeshletVertices[vertexIndex];

        float4 worldPos = entity.Transform * float4(v.Position, 1.0f);

        MSOut vtx;
        vtx.Position      = scene.Camera.ViewProjection * worldPos;
        vtx.UV            = v.UV;
        vtx.Normal        = normalize((entity.Transform * float4(v.Normal, 0.0f)).xyz);
        vtx.WorldPos      = worldPos.xyz;
        vtx.Tangent       = v.Tangent;
        vtx.InstanceIndex = push.InstanceIndex;

        outMesh.set_vertex(gtid, vtx);
    }
}

[[fragment]]
float4 forward_msfs(MSOut in [[stage_in]],
                    const device SceneBuffer& scene [[buffer(0)]]) {
    constexpr sampler textureSampler(
        mag_filter::linear,
        min_filter::linear,
        mip_filter::linear,
        address::repeat,
        lod_clamp(0.0f, MAXFLOAT)
    );

    SceneInstance inst = scene.Instances[in.InstanceIndex];
    SceneMaterial mat  = scene.Materials[inst.MaterialIndex];

    float4 color = float4(1.0f);
    if (mat.hasAlbedo()) {
        color = mat.Albedo.sample(textureSampler, in.UV);
    }

    if (color.a < 0.25f)
        discard_fragment();

    return color;
}
