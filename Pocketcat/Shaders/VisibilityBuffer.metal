//
//  VisibilityBufferVertex.metal
//  Pocketcat
//
//  Created by Amélie Heinrich on 20/03/2026.
//

#include "Common/Bindless.h"

#include <metal_stdlib>
using namespace metal;

#define AS_GROUP_SIZE 32

struct Payload {
    uint InstanceIndex;
};

struct VSOut {
    float4 Position [[position]];
    float2 UV;
    uint InstanceID [[flat]];
    uint MeshletIndex [[flat]];
};

struct FSOut {
    uint2 tri_instance_id;
};

using MeshOutput = mesh<VSOut, void, 64, 128, topology::triangle>;

static inline void alphaTest(VSOut in, const device SceneBuffer& scene) {
    SceneInstance inst = scene.Instances[in.InstanceID];
    SceneMaterial mat  = scene.Materials[inst.MaterialIndex];
    if (mat.AlphaMode == 1 && mat.hasAlbedo()) {
        constexpr sampler s(filter::linear, address::repeat);
        if (mat.Albedo.sample(s, in.UV).a < 0.5) discard_fragment();
    }
}

[[object]]
void visibility_os(const device SceneBuffer& scene [[buffer(0)]],
                   const device uint* instanceIndex [[buffer(1)]],
                   uint gtid [[thread_position_in_threadgroup]],
                   object_data Payload& outPayload [[payload]],
                   mesh_grid_properties outGrid) {
    if (gtid == 0) {
        uint instance = *instanceIndex;
        if (instance >= scene.InstanceCount) return;

        SceneInstance inst = scene.Instances[instance];
        SceneInstanceLOD lod = inst.LODs[0];
        outPayload.InstanceIndex = instance;

        outGrid.set_threadgroups_per_grid(uint3(lod.MeshletCount, 1, 1));
    }
}

[[mesh]]
void visibility_ms(device SceneBuffer& scene [[buffer(0)]],
                   object_data const Payload& payload [[payload]],
                   uint gtid [[thread_position_in_threadgroup]],
                   uint gid [[threadgroup_position_in_grid]],
                   MeshOutput outMesh) {
    uint instanceIndex = payload.InstanceIndex;
    uint meshletIndex = gid;

    SceneInstance inst = scene.Instances[instanceIndex];
    SceneEntity entity = scene.Entities[inst.EntityIndex];
    SceneInstanceLOD lod = inst.LODs[0];

    if (meshletIndex >= lod.MeshletCount) {
        outMesh.set_primitive_count(0);
        return;
    }

    MeshMeshlet m = lod.Meshlets[meshletIndex];
    outMesh.set_primitive_count(m.TriangleCount);

    if (gtid < m.TriangleCount) {
        uint triBase = m.TriangleOffset + gtid * 3;
        outMesh.set_index(gtid * 3 + 0, lod.MeshletTriangles[triBase + 0]);
        outMesh.set_index(gtid * 3 + 1, lod.MeshletTriangles[triBase + 1]);
        outMesh.set_index(gtid * 3 + 2, lod.MeshletTriangles[triBase + 2]);
    }

    if (gtid < m.VertexCount) {
        uint vertexIndex = m.VertexOffset + gtid;
        MeshVertex v = lod.MeshletVertices[vertexIndex];
        float4 worldPos = entity.Transform * float4(v.Position, 1.0f);

        VSOut vtx;
        vtx.Position     = scene.Camera.Projection * scene.Camera.View * worldPos;
        vtx.UV           = v.UV;
        vtx.InstanceID   = instanceIndex;
        vtx.MeshletIndex = meshletIndex;

        outMesh.set_vertex(gtid, vtx);
    }
}

[[vertex]]
VSOut visibility_vs(uint vid [[vertex_id]],
                    const device SceneBuffer& scene [[buffer(0)]],
                    const device uint& instanceID [[buffer(1)]],
                    uint instanceIndex [[base_instance]]) {
    SceneInstance inst = scene.Instances[instanceIndex];
    SceneEntity entity = scene.Entities[inst.EntityIndex];

    MeshVertex v = inst.VertexBuffer[vid];

    float4 worldPos = entity.Transform * float4(v.Position, 1.0f);

    VSOut out;
    out.Position     = scene.Camera.ViewProjection * worldPos;
    out.UV           = v.UV;
    out.InstanceID   = instanceIndex;
    out.MeshletIndex = 0xFFFFFFFF;
    return out;
}

[[fragment]]
FSOut visibility_fs_ms(VSOut in [[stage_in]],
                       uint primID [[primitive_id]],
                       const device SceneBuffer& scene [[buffer(0)]]) {
    alphaTest(in, scene);
    FSOut out;
    out.tri_instance_id = uint2((in.MeshletIndex << 8) | (primID & 0xFF), in.InstanceID);
    return out;
}

[[fragment]]
FSOut visibility_fs_vs(VSOut in [[stage_in]],
                       uint primID [[primitive_id]],
                       const device SceneBuffer& scene [[buffer(0)]]) {
    alphaTest(in, scene);
    FSOut out;
    out.tri_instance_id = uint2(0x80000000u | primID, in.InstanceID);
    return out;
}
