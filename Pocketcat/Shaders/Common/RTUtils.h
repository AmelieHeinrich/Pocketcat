//
//  RTUtils.h
//  Pocketcat
//
//  Created by Amélie Heinrich on 23/03/2026.
//

#ifndef RT_UTILS_H
#define RT_UTILS_H

#include "CookTorrance.h"
#include "Bindless.h"

inline SurfaceHit trace_and_get(
    const device scene_data&                    scene,
    ray                                         r,
    intersector<triangle_data, instancing>     inter)
{
    SurfaceHit result = {};
    result.hit = false;

    auto intersection = inter.intersect(r, scene.tlas);
    if (intersection.type == intersection_type::none)
        return result;

    uint instance_id  = intersection.instance_id;
    uint primitive_id = intersection.primitive_id;
    float2 bary       = intersection.triangle_barycentric_coord;
    float3 ray_dir    = r.direction;

    constexpr sampler s(mag_filter::linear, min_filter::linear,
                        address::repeat, mip_filter::linear);
    
    instance inst = scene.instances[instance_id];
    material mat  = scene.materials[inst.material_index];
    triangle tri  = fetch_triangle(scene, instance_id, primitive_id);

    float2 uv    = interpolate2D(bary, tri.v0.uv,      tri.v1.uv,      tri.v2.uv);
    float3 n_geo = interpolate2D(bary, tri.v0.normal,  tri.v1.normal,  tri.v2.normal);
    float4 t_geo = interpolate2D(bary, tri.v0.tangent, tri.v1.tangent, tri.v2.tangent);

    if (dot(n_geo, -ray_dir) < 0.0) n_geo = -n_geo;

    float4 albedo_sample = mat.has_albedo() ? mat.albedo.sample(s, uv) : float4(0.8, 0.8, 0.8, 1.0);
    float3 orm           = mat.has_orm()    ? mat.orm.sample(s, uv).rgb : float3(1, 0.5, 0);

    float3 n = n_geo;
    if (mat.has_normal()) {
        float3 t   = normalize(t_geo.xyz - dot(t_geo.xyz, n_geo) * n_geo);
        float3 b   = cross(n_geo, t) * t_geo.w;
        float3x3 tbn = float3x3(t, b, n_geo);
        float3 nmap  = mat.normal.sample(s, uv).xyz * 2.0 - 1.0;
        n = normalize(tbn * nmap);
    }

    result.hit      = true;
    result.pos      = r.origin + r.direction * intersection.distance;
    result.n        = n;
    result.albedo   = albedo_sample.rgb;
    result.ao       = orm.r;
    result.roughness = clamp(orm.g, 0.04, 1.0);
    result.metallic  = orm.b;
    result.emissive  = mat.has_emissive() ? mat.emissive.sample(s, uv).rgb : float3(0);
    return result;
}

inline SurfaceHit fetch_secondary_hit(const device scene_data& scene,
                                uint instance_id, uint primitive_id, float2 bary,
                                float3 hit_pos, float3 ray_dir)
{
    constexpr sampler s(mag_filter::linear, min_filter::linear,
                        address::repeat, mip_filter::linear);

    instance  inst = scene.instances[instance_id];
    material  mat  = scene.materials[inst.material_index];
    triangle  tri  = fetch_triangle(scene, instance_id, primitive_id);

    float2 uv    = interpolate2D(bary, tri.v0.uv,      tri.v1.uv,      tri.v2.uv);
    float3 n_geo = interpolate2D(bary, tri.v0.normal,  tri.v1.normal,  tri.v2.normal);
    float4 t_geo = interpolate2D(bary, tri.v0.tangent, tri.v1.tangent, tri.v2.tangent);

    if (dot(n_geo, -ray_dir) < 0.0) n_geo = -n_geo;

    float4 albedo_sample = mat.has_albedo() ? mat.albedo.sample(s, uv) : float4(0.8, 0.8, 0.8, 1.0);
    float3 orm           = mat.has_orm()    ? mat.orm.sample(s, uv).rgb : float3(1, 0.5, 0);

    float3 n = n_geo;
    if (mat.has_normal()) {
        float3 t = normalize(t_geo.xyz - dot(t_geo.xyz, n_geo) * n_geo);
        float3 b = cross(n_geo, t) * t_geo.w;
        float3x3 tbn = float3x3(t, b, n_geo);
        float3 nmap = mat.normal.sample(s, uv).xyz * 2.0 - 1.0;
        n = normalize(tbn * nmap);
    }

    SurfaceHit h;
    h.pos       = hit_pos;
    h.n         = n;
    h.albedo    = albedo_sample.rgb;
    h.ao        = orm.r;
    h.roughness = clamp(orm.g, 0.04, 1.0);
    h.metallic  = orm.b;
    h.emissive  = mat.has_emissive() ? mat.emissive.sample(s, uv).rgb : float3(0);
    return h;
}

inline float visibility(float3 origin, float3 dir, float max_dist,
                 const device scene_data& scene,
                 intersection_function_table<triangle_data, instancing> ift)
{
    ray r;
    r.origin = origin;
    r.direction = dir;
    r.min_distance = 0.001;
    r.max_distance = max_dist;

    intersector<triangle_data, instancing> si;
    si.assume_geometry_type(geometry_type::triangle);
    si.accept_any_intersection(true);

    return (si.intersect(r, scene.tlas, 0xFF, ift).type == intersection_type::none) ? 1.0 : 0.0;
}

inline float visibility(float3 origin, float3 dir, float max_dist,
                 const device scene_data& scene)
{
    ray r;
    r.origin = origin;
    r.direction = dir;
    r.min_distance = 0.001;
    r.max_distance = max_dist;

    intersector<triangle_data, instancing> si;
    si.assume_geometry_type(geometry_type::triangle);
    si.accept_any_intersection(true);

    return (si.intersect(r, scene.tlas, 0xFF).type == intersection_type::none) ? 1.0 : 0.0;
}

#endif
