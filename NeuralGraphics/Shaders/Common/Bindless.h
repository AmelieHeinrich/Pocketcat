//
//  Bindless.h
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 02/03/2026.
//

#ifndef BINDLESS_METAL_H
#define BINDLESS_METAL_H

#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
using namespace raytracing;

constant constexpr uint kMaxLODs = 5;

struct MeshVertex
{
    packed_float3 Position;
    packed_float3 Normal;
    float2        UV;
    float4        Tangent;
};

struct MeshMeshlet
{
    uint VertexOffset;
    uint TriangleOffset;
    uint VertexCount;
    uint TriangleCount;
};

struct MeshMeshletBounds
{
    float3   Center;
    float    Radius;
    float3   ConeApex;
    float3   ConeAxis;
    float    ConeCutoff;
    int8_t   ConeAxisS8[3];
    int8_t   ConeCutoffS8;
};

enum SceneMaterialFlags : uint
{
    MaterialFlag_HasAlbedo   = (1 << 0),
    MaterialFlag_HasNormal   = (1 << 1),
    MaterialFlag_HasORM      = (1 << 2),
    MaterialFlag_HasEmissive = (1 << 3),
    MaterialFlag_IsOpaque    = (1 << 4),
};

struct SceneMaterial
{
    texture2d<float> Albedo;
    texture2d<float> Normal;
    texture2d<float> ORM;
    texture2d<float> Emissive;
    uint Flags;
    uint AlphaMode;

    bool hasAlbedo()   const { return Flags & MaterialFlag_HasAlbedo;   }
    bool hasNormal()   const { return Flags & MaterialFlag_HasNormal;   }
    bool hasORM()      const { return Flags & MaterialFlag_HasORM;      }
    bool hasEmissive() const { return Flags & MaterialFlag_HasEmissive; }
    bool isOpaque()    const { return Flags & MaterialFlag_IsOpaque;    }
};

struct SceneInstanceLOD
{
    const device uint*              IndexBuffer;
    const device MeshMeshlet*       Meshlets;
    const device MeshVertex*        MeshletVertices;
    const device uchar*             MeshletTriangles;
    const device MeshMeshletBounds* MeshletBounds;
    uint IndexCount;
    uint MeshletCount;
};

struct SceneInstance
{
    const device MeshVertex* VertexBuffer;
    MTLResourceID blas;
    uint MaterialIndex;
    uint EntityIndex;
    uint LODCount;
    float3 AABBMin;
    float3 AABBMax;
    SceneInstanceLOD LODs[kMaxLODs];
};

struct SceneEntity
{
    float4x4 Transform;
};

struct SceneCamera
{
    float4x4 View;
    float4x4 Projection;
    float4x4 ViewProjection;
    float4x4 InverseView;
    float4x4 InverseProjection;
    float4x4 InverseViewProjection;
    float4   PositionAndNear;   // .xyz = position, .w = near
    float4   DirectionAndFar;   // .xyz = direction, .w = far

    float3 GetPosition()  const { return PositionAndNear.xyz;  }
    float  GetNear()      const { return PositionAndNear.w;    }
    float3 GetDirection() const { return DirectionAndFar.xyz;  }
    float  GetFar()       const { return DirectionAndFar.w;    }
};

struct DebugVertex
{
    packed_float3 Position;
    packed_float4 Color;
};

struct SceneBuffer
{
    const device SceneMaterial* Materials;
    const device SceneInstance* Instances;
    const device SceneEntity*   Entities;
    instance_acceleration_structure AccelerationStructure;

    SceneCamera Camera;
    uint MaterialCount;
    uint InstanceCount;
    uint EntityCount;

    device DebugVertex* DebugVertices;
    device atomic_uint* DebugVertexCount;
    uint                MaxDebugVertices;
};

inline float3 HashColor(uint id)
{
    uint h = id;
    h ^= h >> 16;
    h *= 0x45d9f3b;
    h ^= h >> 16;

    float hue = float(h & 0xFFFF) / 65535.0;
    hue = fmod(hue + 0.33, 1.0);

    float s = 1.0;
    float v = 1.0;

    float3 rgb = clamp(abs(fmod(hue * 6.0 + float3(0, 4, 2), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    return v * mix(float3(1, 1, 1), rgb, s);
}

#endif
