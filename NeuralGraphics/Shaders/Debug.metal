//
//  Debug.metal
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 03/03/2026.
//

#include <metal_stdlib>
using namespace metal;

#include "Common/Bindless.h"

struct DebugVSOut {
    float4 Position [[position]];
    float4 Color;
};

struct DebugData {
    float4x4 Camera;
};

struct DebugICBWrapper {
    command_buffer CommandBuffer;
};

[[vertex]]
DebugVSOut debug_vs(uint id [[vertex_id]],
                    const device DebugData&    data     [[buffer(0)]],
                    const device DebugVertex*  vertices [[buffer(1)]]) {
    DebugVSOut out;
    out.Position = data.Camera * float4(vertices[id].Position, 1.0f);
    out.Color    = vertices[id].Color;
    return out;
}

[[fragment]]
float4 debug_fs(DebugVSOut in [[stage_in]]) {
    return in.Color;
}

[[kernel]]
void debug_generate_icb(device SceneBuffer* scene [[buffer(0)]],
                        device DebugICBWrapper& icb [[buffer(1)]],
                        uint threadID [[thread_position_in_grid]]) {
    if (threadID > 0) return;

    uint vertexCount = atomic_load_explicit(scene->DebugVertexCount, memory_order_relaxed);

    render_command cmd(icb.CommandBuffer, threadID);
    if (vertexCount > 0) {
        // Bind the GPU debug vertex buffer and generate a single draw call
        cmd.set_vertex_buffer(scene->DebugVertices, 1);
        cmd.draw_primitives(primitive_type::line, 0, vertexCount, 1, 0);
    }
}
