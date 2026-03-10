//
//  MeshCompressor.h
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 02/03/2026.
//

#pragma once

#include <string>
#include <vector>

#define SKIP_TEXTURES 0

// Maximum number of LOD levels exported per primitive.
// LOD0 = full detail, LOD1..kMaxLODs-1 = progressively simplified.
// Simple meshes (e.g. cubes) may have fewer LODs if simplification
// produces no meaningful reduction; the actual count is stored in MeshHeader.
static constexpr uint32_t kMaxLODs = 5; // LOD0 through LOD4

// Simplification thresholds per LOD level (fraction of original index count).
// LOD0 is always 1.0 (full detail).
static constexpr float kLODThresholds[kMaxLODs] = {
    1.0f,   // LOD0: full detail
    0.5f,   // LOD1: 50%
    0.25f,  // LOD2: 25%
    0.125f, // LOD3: 12.5%
    0.0625f // LOD4: 6.25%
};

// Maximum allowed simplification error per LOD level.
static constexpr float kLODTargetErrors[kMaxLODs] = {
    0.0f,   // LOD0: no error (not simplified)
    1e-2f,  // LOD1
    2e-2f,  // LOD2
    5e-2f,  // LOD3
    1e-1f   // LOD4
};

struct MeshVertex
{
    float Position[3];
    float Normal[3];
    float UV[2];
    float Tangent[4];
};

struct MeshMeshlet
{
    uint32_t VertexOffset;
    uint32_t TriangleOffset;
    uint32_t VertexCount;
    uint32_t TriangleCount;
};

struct MeshInstance
{
    float AABBMin[3];
    float AABBMax[3];
    uint32_t VertexOffset;
    uint32_t MaterialIndex;

    // Per-LOD offsets and counts.
    // Only the first header.LODCount entries are valid.
    uint32_t IndexOffset[kMaxLODs];
    uint32_t IndexCount[kMaxLODs];
    uint32_t MeshletOffset[kMaxLODs];
    uint32_t MeshletVerticesOffset[kMaxLODs];
    uint32_t MeshletIndicesOffset[kMaxLODs];
    uint32_t MeshletBoundsOffset[kMaxLODs];
    uint32_t MeshletCount[kMaxLODs];
};

struct MeshMaterial
{
    char AlbedoPath[256];
    char NormalPath[256];
    char ORMPath[256];
    char EmissivePath[256];
    uint32_t AlphaMode; // 0 = opaque, 1 = mask, 2 = blend
};

struct MeshMeshletBounds
{
    float    Center[3];
    float    Radius;
    float    ConeApex[3];
    float    ConeAxis[3];
    float    ConeCutoff;
    int8_t   ConeAxisS8[3];
    int8_t   ConeCutoffS8;
};

struct MeshHeader
{
    uint32_t InstanceCount;
    uint32_t MaterialCount;
    uint32_t LODCount;       // Number of LOD levels present (1..kMaxLODs)
};

void CompressMesh(const std::string& in, const std::string& out);
