//
//  RTGI.metal
//  Pocketcat
//
//  Created by Amélie Heinrich on 23/03/2026.
//

#include "Common/Bindless.h"
#include "Common/Math.h"
#include "Common/RNG.h"
#include "Common/CookTorrance.h"
#include "Common/RTUtils.h"

struct rtgi_parameters {
    uint frame_id;
    uint spp;
    float resolution_scale;
    uint padding;
};

[[kernel]]
void rtgi(texture2d<float, access::read_write> out [[texture(0)]],
          texture2d<float> depth_texture [[texture(1)]],
          texture2d<float> normal_texture [[texture(2)]],
          texture2d<float> albedo_texture [[texture(3)]],
          const device scene_data& scene [[buffer(0)]],
          const device rtgi_parameters& parameters [[buffer(1)]],
          uint2 pixel_id [[thread_position_in_grid]])
{
    // For this technique specifically we don't do alpha testing, not worth it
    uint width = out.get_width();
    uint height = out.get_height();
    if (pixel_id.x >= width || pixel_id.y >= height)
        return;

    float2 dimensions = float2(width, height);
    float2 pixel_center = float2(pixel_id) + 0.5;
    float2 uv = pixel_center / dimensions;
    float2 ndc = uv * 2.0 - 1.0;
    ndc.y = -ndc.y;

    uint2 read_pixel_id = uint2(float2(pixel_id) / parameters.resolution_scale);

    float depth = depth_texture.read(read_pixel_id).x;
    float3 n = normal_texture.read(read_pixel_id).rgb;
    float4 clip4 = float4(ndc, depth, 1.0);
    float4 world4 = scene.camera.inverse_view_projection * clip4;
    float3 world_pos = world4.xyz / world4.w;
    float3 albedo = albedo_texture.read(read_pixel_id).rgb;

    RNG rng = make_rng(pixel_id, parameters.frame_id);

    intersector<triangle_data, instancing> inter;
    inter.assume_geometry_type(geometry_type::triangle);

    const float3 light_dir = scene.sun.direction_and_radius.xyz;
    const float3 light_color = scene.sun.color_and_intensity.xyz * scene.sun.color_and_intensity.w;

    float3 indirect = 0.0;
    for (uint i = 0; i < parameters.spp; i++) {
        float3 wi = sample_cosine_hemisphere(n, rng.next_f(), rng.next_f());

        ray ray;
        ray.direction = wi;
        ray.origin = world_pos + n * 0.001;
        ray.min_distance = 0.001;
        ray.max_distance = 1000;

        SurfaceHit bounce = trace_and_get(scene, ray, inter);
        if (!bounce.hit) {
            continue;
        }
        float3 li = eval_brdf(bounce, -wi, -light_dir) * visibility(bounce.pos + bounce.n * 0.001, -light_dir, 1000, scene) * light_color;
        indirect += albedo * (bounce.emissive + li);
    }
    indirect /= parameters.spp;

    out.write(float4(indirect, 1.0), pixel_id);
}
