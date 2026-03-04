//
//  ModelMS.swift
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 03/03/2026.
//

#include <metal_stdlib>
using namespace metal;

struct Meshlet {
    uint VertexOffset;
    uint TriangleOffset;
    uint VertexCount;
    uint TriangleCount;
};

struct VSIn {
    packed_float3 Position;
    packed_float3 Normal;
    float2        UV;
    float4        Tangent;
};

struct VSOut {
    float4 Position [[position]];
    float3 Color;
};

struct ModelData {
    float4x4 Camera;
    uint VertexOffset;
};

using MeshOutput = metal::mesh<VSOut, void, 64, 128, topology::triangle>;

uint hash(uint a)
{
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

[[mesh]]
void forward_ms(const device ModelData& modelData [[buffer(0)]],
                device const VSIn* vertices [[buffer(1)]],
                device const Meshlet* meshlets [[buffer(2)]],
                device const uint* meshletVertices [[buffer(3)]],
                device const uchar* meshletIndices [[buffer(4)]],
                uint gtid [[thread_position_in_threadgroup]],
                uint gid [[threadgroup_position_in_grid]],
                MeshOutput outMesh) {

    device const Meshlet& m = meshlets[gid];
    outMesh.set_primitive_count(m.TriangleCount);

    if (gtid < m.TriangleCount) {
        uint triBase = m.TriangleOffset + gtid * 3;
        uint vIdx0   = meshletIndices[triBase + 0];
        uint vIdx1   = meshletIndices[triBase + 1];
        uint vIdx2   = meshletIndices[triBase + 2];

        uint triIdx = 3 * gtid;
        outMesh.set_index(triIdx + 0, vIdx0);
        outMesh.set_index(triIdx + 1, vIdx1);
        outMesh.set_index(triIdx + 2, vIdx2);
    }

    if (gtid < m.VertexCount) {
        uint vertexIndex = m.VertexOffset + gtid;
        vertexIndex = meshletVertices[vertexIndex];

        uint meshletHash = hash(gid);
        float3 meshletColor = float3(float(meshletHash & 255), float((meshletHash >> 8) & 255), float((meshletHash >> 16) & 255)) / 255.0;

        VSOut vtx;
        vtx.Position = modelData.Camera * float4(vertices[vertexIndex].Position, 1.0);
        vtx.Color = meshletColor;

        outMesh.set_vertex(gtid, vtx);
    }
}

[[fragment]]
float4 forward_msfs(VSOut in [[stage_in]]) {
    return float4(in.Color, 1.0);
}
