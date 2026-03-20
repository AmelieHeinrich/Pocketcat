//
//  MeshPathCull.metal
//  Pocketcat
//
//  Created by Amélie Heinrich on 08/03/2026.
//

#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
using namespace simd;

#include "Common/Bindless.h"
#include "Common/DebugDraw.h"

struct ICBWrapper {
    command_buffer CommandBuffer;
};

[[kernel]]
void vertex_geometry_cull(const device SceneBuffer* scene [[buffer(0)]],
                          device ICBWrapper& icb [[buffer(1)]],
                          constant uint& instanceCount [[buffer(2)]],
                          uint threadID [[thread_position_in_grid]]) {
    uint instanceIndex = threadID;
    if (instanceIndex >= instanceCount) return;

    SceneInstance inst = scene->Instances[instanceIndex];
    SceneInstanceLOD lod = inst.LODs[0];

    bool visible = true;
    if (visible) {
        render_command command(icb.CommandBuffer, instanceIndex);
        command.reset();
        command.set_vertex_buffer(scene, 0);
        command.set_fragment_buffer(scene, 0);
        command.draw_indexed_primitives(primitive_type::triangle, lod.IndexCount, lod.IndexBuffer, 1, 0, instanceIndex);
    }
}

[[kernel]]
void mesh_geometry_cull(device SceneBuffer* scene [[buffer(0)]],
                        device ICBWrapper& icb [[buffer(1)]],
                        constant uint& instanceCount [[buffer(2)]],
                        device uint* instanceIDs [[buffer(3)]],
                        uint threadID [[thread_position_in_grid]]) {
    if (threadID >= instanceCount) return;

    bool visible = true;
    if (visible) {
        instanceIDs[threadID] = threadID;

        render_command command(icb.CommandBuffer, threadID);
        command.set_object_buffer(scene, 0);
        command.set_object_buffer(instanceIDs + threadID, 1);
        command.set_mesh_buffer(scene, 0);
        command.set_fragment_buffer(scene, 0);
        command.draw_mesh_threadgroups(
            uint3(1, 1, 1),             // 1 Object group
            uint3(32, 1, 1),            // threadsPerObjectThreadgroup
            uint3(128, 1, 1)            // threadsPerMeshThreadgroup
        );
    }
}
