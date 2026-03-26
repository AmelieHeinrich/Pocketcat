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

struct payload {
    uint instance_index;
};

struct vs_out {
    float4 position [[position]];
    float4 curr_clip_pos;
    float4 prev_clip_pos;
    float2 uv;
    uint instance_id [[flat]];
    uint meshlet_index [[flat]];
};

struct fs_out {
    uint2  tri_instance_id [[color(0)]];
    float3 motion_vector   [[color(1)]];
};

using mesh_output = mesh<vs_out, void, 64, 128, topology::triangle>;

static inline void alpha_test(vs_out in, const device scene_data& scene) {
    instance inst = scene.instances[in.instance_id];
    material mat  = scene.materials[inst.material_index];
    if (mat.alpha_mode != 0 && mat.has_albedo()) {
        constexpr sampler s(filter::linear, address::repeat);
        if (mat.albedo.sample(s, in.uv).a < 0.25) discard_fragment();
    }
}

[[object]]
void visibility_os(const device scene_data& scene [[buffer(0)]],
                   const device uint* instance_index [[buffer(1)]],
                   uint gtid [[thread_position_in_threadgroup]],
                   object_data payload& out_payload [[payload]],
                   mesh_grid_properties out_grid) {
    if (gtid == 0) {
        uint instance_id = *instance_index;
        if (instance_id >= scene.instance_count) return;

        instance inst = scene.instances[instance_id];
        instance_lod lod = inst.lods[0];
        out_payload.instance_index = instance_id;

        out_grid.set_threadgroups_per_grid(uint3(lod.meshlet_count, 1, 1));
    }
}

[[mesh]]
void visibility_ms(device scene_data& scene [[buffer(0)]],
                   object_data const payload& payload [[payload]],
                   uint gtid [[thread_position_in_threadgroup]],
                   uint gid [[threadgroup_position_in_grid]],
                   mesh_output out_mesh) {
    uint instance_index = payload.instance_index;
    uint meshlet_idx = gid;

    instance inst = scene.instances[instance_index];
    entity entity = scene.entities[inst.entity_index];
    instance_lod lod = inst.lods[0];

    if (meshlet_idx >= lod.meshlet_count) {
        out_mesh.set_primitive_count(0);
        return;
    }

    meshlet m = lod.meshlets[meshlet_idx];
    out_mesh.set_primitive_count(m.triangle_count);

    if (gtid < m.triangle_count) {
        uint tri_base = m.triangle_offset + gtid * 3;
        out_mesh.set_index(gtid * 3 + 0, lod.meshlet_triangles[tri_base + 0]);
        out_mesh.set_index(gtid * 3 + 1, lod.meshlet_triangles[tri_base + 1]);
        out_mesh.set_index(gtid * 3 + 2, lod.meshlet_triangles[tri_base + 2]);
    }

    if (gtid < m.vertex_count) {
        uint vertex_idx = m.vertex_offset + gtid;
        mesh_vertex v = lod.meshlet_vertices[vertex_idx];
        float4 world_pos = entity.transform * float4(v.position, 1.0f);

        vs_out vtx;
        vtx.position      = scene.camera.view_projection * world_pos;
        vtx.curr_clip_pos = scene.camera.view_projection_no_jitter * world_pos;
        vtx.prev_clip_pos = scene.camera.previous_view_projection * world_pos;
        vtx.uv            = v.uv;
        vtx.instance_id   = instance_index;
        vtx.meshlet_index = meshlet_idx;

        out_mesh.set_vertex(gtid, vtx);
    }
}

[[vertex]]
vs_out visibility_vs(uint vid [[vertex_id]],
                     const device scene_data& scene [[buffer(0)]],
                     uint instance_index [[base_instance]]) {
    instance inst = scene.instances[instance_index];
    entity entity = scene.entities[inst.entity_index];

    mesh_vertex v = inst.vertex_buffer[vid];

    float4 world_pos = entity.transform * float4(v.position, 1.0f);
    float4 clip_pos  = scene.camera.view_projection * world_pos;

    vs_out out;
    out.position      = clip_pos;
    out.curr_clip_pos = scene.camera.view_projection_no_jitter * world_pos;
    out.prev_clip_pos = scene.camera.previous_view_projection * world_pos;
    out.uv            = v.uv;
    out.instance_id   = instance_index;
    out.meshlet_index = 0xFFFFFFFF;
    return out;
}

static inline float3 compute_motion_vector(vs_out in) {
    float2 curr_ndc = in.curr_clip_pos.xy / in.curr_clip_pos.w;
    float2 prev_ndc = in.prev_clip_pos.xy / in.prev_clip_pos.w;
    float2 motion   = (prev_ndc - curr_ndc) * float2(0.5f, -0.5f);
    float  depth    = in.curr_clip_pos.w;
    return float3(motion, depth);
}

[[fragment]]
fs_out visibility_fs_ms(vs_out in [[stage_in]],
                        uint prim_id [[primitive_id]],
                        const device scene_data& scene [[buffer(0)]]) {
    alpha_test(in, scene);
    fs_out out;
    out.tri_instance_id = uint2((in.meshlet_index << 8) | (prim_id & 0xFF), in.instance_id);
    out.motion_vector   = compute_motion_vector(in);
    return out;
}

[[fragment]]
fs_out visibility_fs_vs(vs_out in [[stage_in]],
                        uint prim_id [[primitive_id]],
                        const device scene_data& scene [[buffer(0)]]) {
    alpha_test(in, scene);
    fs_out out;
    out.tri_instance_id = uint2(0x80000000u | prim_id, in.instance_id);
    out.motion_vector   = compute_motion_vector(in);
    return out;
}
