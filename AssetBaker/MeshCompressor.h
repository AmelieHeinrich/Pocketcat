//
//  MeshCompressor.h
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 02/03/2026.
//

#pragma once

#include <string>
#include <vector>

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
    uint32_t IndexOffset;
    uint32_t MeshletOffset;
    uint32_t MeshletVerticesOffset;
    uint32_t MeshletIndicesOffset;
    uint32_t MeshletBoundsOffset;
    uint32_t MaterialIndex;
};

struct MeshMaterial
{
    char AlbedoPath[256];
    char NormalPath[256];
    char ORMPath[256];
    char EmissivePath[256];
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
};

void CompressMesh(const std::string& in, const std::string& out);
