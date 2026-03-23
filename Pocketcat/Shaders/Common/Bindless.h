//
//  Bindless.h
//  Pocketcat
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

struct mesh_vertex
{
    packed_float3 position;
    packed_float3 normal;
    float2        uv;
    float4        tangent;
};

struct meshlet
{
    uint vertex_offset;
    uint triangle_offset;
    uint vertex_count;
    uint triangle_count;
};

struct meshlet_bounds
{
    float3   center;
    float    radius;
    float3   cone_apex;
    float3   cone_axis;
    float    cone_cutoff;
    int8_t   cone_axisS8[3];
    int8_t   cone_cutoffS8;
};

enum material_flags : uint
{
    MaterialFlag_HasAlbedo   = (1 << 0),
    MaterialFlag_HasNormal   = (1 << 1),
    MaterialFlag_HasORM      = (1 << 2),
    MaterialFlag_HasEmissive = (1 << 3),
    MaterialFlag_IsOpaque    = (1 << 4),
};

struct material
{
    texture2d<float> albedo;
    texture2d<float> normal;
    texture2d<float> orm;
    texture2d<float> emissive;
    uint flags;
    uint alpha_mode;

    bool has_albedo()   const { return flags & MaterialFlag_HasAlbedo;   }
    bool has_normal()   const { return flags & MaterialFlag_HasNormal;   }
    bool has_orm()      const { return flags & MaterialFlag_HasORM;      }
    bool has_emissive() const { return flags & MaterialFlag_HasEmissive; }
    bool is_opaque()    const { return flags & MaterialFlag_IsOpaque;    }
};

struct instance_lod
{
    const device uint* index_buffer;
    const device meshlet* meshlets;
    const device mesh_vertex* meshlet_vertices;
    const device uchar* meshlet_triangles;
    const device meshlet_bounds* meshlet_bounds;
    uint index_count;
    uint meshlet_count;
};

struct instance
{
    const device mesh_vertex* vertex_buffer;
    MTLResourceID blas;
    uint material_index;
    uint entity_index;
    uint lod_count;
    float3 aabb_min;
    float3 aabb_max;
    instance_lod lods[kMaxLODs];
};

struct entity
{
    float4x4 transform;
};

struct camera
{
    float4x4 view;
    float4x4 projection;
    float4x4 view_projection;
    float4x4 view_projection_no_jitter;
    float4x4 inverse_view;
    float4x4 inverse_projection;
    float4x4 inverse_view_projection;
    float4x4 previous_view_projection;
    float4   position_and_near;   // .xyz = position, .w = near
    float4   direction_and_far;   // .xyz = direction, .w = far

    const float3 get_position()  const { return position_and_near.xyz;  }
    const float  get_near()      const { return position_and_near.w;    }
    const float3 get_direction() const { return direction_and_far.xyz;  }
    const float  get_far()       const { return direction_and_far.w;    }
};

struct debug_vertex
{
    packed_float3 position;
    packed_float4 color;
};

struct sun_light
{
    float4 direction_and_radius;   // xyz = direction (normalized toward light), w = angular radius
    float4 color_and_intensity;    // xyz = color, w = intensity
};

struct point_light
{
    float4 position_and_radius;   // xyz = world position, w = influence radius
    float4 color_and_intensity;   // xyz = color, w = intensity
};

struct scene_data
{
    const device material* materials;
    const device instance* instances;
    const device entity*   entities;
    instance_acceleration_structure tlas;

    camera camera;
    uint material_count;
    uint instance_count;
    uint entity_count;

    device debug_vertex* debug_vertices;
    device atomic_uint* debug_vertex_count;
    uint max_debug_vertices;

    uint _pad_lights;
    const device point_light* point_lights;
    uint point_light_count;
    uint _pad_lights2;
    sun_light sun;
};

struct triangle {
    mesh_vertex v0;
    mesh_vertex v1;
    mesh_vertex v2;
};

inline triangle fetch_triangle_encoded(const device scene_data& scene, uint draw_id, uint encoded_prim_id) {
    instance instance = scene.instances[draw_id];
    instance_lod lod = instance.lods[0];

    triangle tri;
    if (encoded_prim_id & 0x80000000u) {
        uint prim_id = encoded_prim_id & 0x7FFFFFFFu;
        uint base = prim_id * 3;
        uint i0 = lod.index_buffer[base];
        uint i1 = lod.index_buffer[base + 1];
        uint i2 = lod.index_buffer[base + 2];
        tri.v0 = instance.vertex_buffer[i0];
        tri.v1 = instance.vertex_buffer[i1];
        tri.v2 = instance.vertex_buffer[i2];
    } else {
        uint meshlet_index = encoded_prim_id >> 8;
        uint local_tri = encoded_prim_id & 0xFF;
        meshlet m = lod.meshlets[meshlet_index];
        uint tri_base = m.triangle_offset + local_tri * 3;
        uint lv0 = lod.meshlet_triangles[tri_base + 0];
        uint lv1 = lod.meshlet_triangles[tri_base + 1];
        uint lv2 = lod.meshlet_triangles[tri_base + 2];
        tri.v0 = lod.meshlet_vertices[m.vertex_offset + lv0];
        tri.v1 = lod.meshlet_vertices[m.vertex_offset + lv1];
        tri.v2 = lod.meshlet_vertices[m.vertex_offset + lv2];
    }
    return tri;
}

inline triangle fetch_triangle(const device scene_data& scene, uint draw_id, uint prim_id) {
    instance instance = scene.instances[draw_id];
    instance_lod lod = instance.lods[0];
    triangle tri;
    
    uint base = prim_id * 3;
    uint i0 = lod.index_buffer[base];
    uint i1 = lod.index_buffer[base + 1];
    uint i2 = lod.index_buffer[base + 2];
    tri.v0 = instance.vertex_buffer[i0];
    tri.v1 = instance.vertex_buffer[i1];
    tri.v2 = instance.vertex_buffer[i2];
    return tri;
}

#endif
