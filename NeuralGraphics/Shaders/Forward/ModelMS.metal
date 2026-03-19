//
//  ModelMS.metal
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 03/03/2026.
//

#include "../Common/Bindless.h"
#include "../Common/DebugDraw.h"

#include <metal_stdlib>
using namespace metal;

#define AS_GROUP_SIZE 32

struct Payload {
    uint InstanceIndex;
};

struct MSOut {
    float4 Position [[position]];
    float2 UV;
    float3 Normal;
    float3 WorldPos;
    float4 Tangent;
    float3 MeshletColor;
    uint MaterialIndex [[flat]];
};

struct MSTriOut {
    bool Culled [[primitive_culled]];
};

using MeshOutput = mesh<MSOut, MSTriOut, 64, 128, topology::triangle>;

[[object]]
void forward_os(const device SceneBuffer& scene [[buffer(0)]],
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
void forward_ms(device SceneBuffer& scene [[buffer(0)]],
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

    // --- Index Processing ---
    if (gtid < m.TriangleCount) {
        uint triBase = m.TriangleOffset + gtid * 3;
        outMesh.set_index(gtid * 3 + 0, lod.MeshletTriangles[triBase + 0]);
        outMesh.set_index(gtid * 3 + 1, lod.MeshletTriangles[triBase + 1]);
        outMesh.set_index(gtid * 3 + 2, lod.MeshletTriangles[triBase + 2]);
        
        MSTriOut tri;
        tri.Culled = false;
        
        outMesh.set_primitive(gtid, tri);
    }

    // --- Vertex Processing ---
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
        vtx.MaterialIndex = inst.MaterialIndex;
        vtx.MeshletColor = HashColor(meshletIndex);

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

    SceneMaterial mat  = scene.Materials[in.MaterialIndex];

    float4 color = float4(1.0f);
    if (mat.hasAlbedo()) {
        color = mat.Albedo.sample(textureSampler, in.UV);
    }

    if (color.a < 0.25f)
        discard_fragment();
    
    intersector<triangle_data, instancing> i;
    i.accept_any_intersection(true);
    i.assume_geometry_type(geometry_type::triangle);
    i.force_opacity(forced_opacity::opaque);
    
    ray ray;
    ray.direction = -float3(0.0, -0.8, 0.22);
    ray.origin = in.WorldPos + in.Normal * 0.005;
    ray.min_distance = 0.001;
    ray.max_distance = 1000;
    
    typename intersector<triangle_data, instancing>::result_type result;
    result = i.intersect(ray, scene.AccelerationStructure, 0xFF);
    bool occluded = (result.type == intersection_type::none);
    
    return float4(in.MeshletColor * occluded, 1.0);
}
