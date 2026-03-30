//
//  RTShadows.metal
//  Pocketcat
//
//  Created by Amélie Heinrich on 23/03/2026.
//

#include "Common/Bindless.h"
#include "Common/Math.h"
#include "Common/RNG.h"
#include "Common/CookTorrance.h"
#include "Common/RTUtils.h"

struct rt_shadow_parameters {
    uint frame_id;
    uint spp;
};

[[kernel]]
void rt_shadows(texture2d<float, access::write> out [[texture(0)]],
                texture2d<float> depth_texture [[texture(1)]],
                texture2d<float> normal_texture [[texture(2)]],
                const device scene_data& scene [[buffer(0)]],
                const device rt_shadow_parameters& parameters [[buffer(1)]],
                intersection_function_table<triangle_data, instancing> ift [[buffer(2)]],
                uint2 pixel_id [[thread_position_in_grid]])
{
    uint width = out.get_width();
    uint height = out.get_height();
    if (pixel_id.x >= width || pixel_id.y >= height)
        return;
    
    const float3 light_dir = -scene.sun.direction_and_radius.xyz;
    const float light_radius = radians(scene.sun.direction_and_radius.w);
    const float3 safe_up = abs(light_dir.y) > 0.999f ? float3(1.0f, 0.0f, 0.0f) : float3(0.0f, 1.0f, 0.0f);
    const float3 light_tangent = normalize(cross(safe_up, light_dir));
    const float3 light_bitangent = normalize(cross(light_dir, light_tangent));

    float2 dimensions = float2(width, height);
    float2 pixel_center = float2(pixel_id) + 0.5;
    float2 uv = pixel_center / dimensions;
    float2 ndc = uv * 2.0 - 1.0;
    ndc.y = -ndc.y;

    float depth = depth_texture.read(pixel_id).x;
    float3 n = normal_texture.read(pixel_id).rgb;
    if (dot(n, light_dir) <= 0.0) {
        out.write(0, pixel_id);
        return;
    }
    
    float4 clip4 = float4(ndc, depth, 1.0);
    float4 world4 = scene.camera.inverse_view_projection * clip4;
    float3 world_pos = world4.xyz / world4.w;

    RNG rng = make_rng(pixel_id, parameters.frame_id);

    intersector<triangle_data, instancing> inter;
    inter.assume_geometry_type(geometry_type::triangle);
    inter.accept_any_intersection(true);
    
    float visibility = 0.0;
    for (uint i = 0; i < parameters.spp; i++) {
        float r1 = rng.next_f();
        float r2 = rng.next_f();
        float point_radius = light_radius * sqrt(r1);
        float point_angle = r2 * 2.0f * M_PI_F;
        float2 disk_point = float2(point_radius * cos(point_angle), point_radius * sin(point_angle));
        
        float3 wi = normalize(light_dir + disk_point.x * light_tangent + disk_point.y * light_bitangent);
        
        ray ray;
        ray.origin = world_pos + n * 0.005f;
        ray.direction = wi;
        ray.min_distance = 0.001f;
        ray.max_distance = 300.0f;
        
        auto result = inter.intersect(ray, scene.tlas, 0xFF, ift);
        visibility += (result.type == intersection_type::none);
    }
    visibility /= parameters.spp;

    out.write(visibility, pixel_id);
}
