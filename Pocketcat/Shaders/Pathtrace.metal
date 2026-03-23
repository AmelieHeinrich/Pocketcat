//
//  PrimaryRayTest.metal
//  Pocketcat
//
//  Created by Amélie Heinrich on 20/03/2026.
//

#include "Common/Bindless.h"
#include "Common/Math.h"
#include "Common/PBR.h"
#include "Common/RNG.h"
#include "Common/CookTorrance.h"
#include "Common/RTUtils.h"

struct pathtracer_parameters {
    uint spp;
    uint bounce_count;
    uint frame_index;
};

// ─── Pathtracer ───────────────────────────────────────────────────────────────

[[kernel]]
void pathtracer(const device scene_data& scene [[buffer(0)]],
                intersection_function_table<triangle_data, instancing> ift [[buffer(1)]],
                constant pathtracer_parameters& parameters [[buffer(2)]],
                texture2d<float> depth_texture           [[texture(0)]],
                texture2d<float> albedo_texture          [[texture(1)]],
                texture2d<float> normal_texture          [[texture(2)]],
                texture2d<float> orm_texture             [[texture(3)]],
                texture2d<float> emissive_texture        [[texture(4)]],
                texture2d<float, access::write> output   [[texture(5)]],
                uint2 pid [[thread_position_in_grid]])
{
    const float3 light_dir = scene.sun.direction_and_radius.xyz;
    const float3 light_color = scene.sun.color_and_intensity.xyz * scene.sun.color_and_intensity.w;

    uint width = output.get_width();
    uint height = output.get_height();
    if (pid.x >= width || pid.y >= height) return;

    float2 dimensions = float2(width, height);
    float2 pixel_center = float2(pid) + 0.5;
    float2 uv = pixel_center / dimensions;
    float2 ndc = uv * 2.0 - 1.0;
    ndc.y = -ndc.y;

    // Sky

    float depth = depth_texture.read(pid).r;
    if (depth >= 1.0) {
        output.write(0.0, pid);
        return;
    }

    // Reconstruct

    float4 clip4     = float4(ndc, depth, 1.0);
    float4 world4    = scene.camera.inverse_view_projection * clip4;
    float3 world_pos = world4.xyz / world4.w;

    SurfaceHit current_hit;
    current_hit.pos = world_pos;
    current_hit.n = normalize(normal_texture.read(pid).xyz);
    current_hit.albedo = albedo_texture.read(pid).rgb;
    float2 rm = orm_texture.read(pid).rg;
    current_hit.roughness = max(rm.r, 0.04);
    current_hit.metallic  = rm.g;
    current_hit.ao = 1.0;
    current_hit.emissive  = emissive_texture.read(pid).rgb;

    float3 cam_pos = scene.camera.position_and_near.xyz;
    float3 v0      = normalize(cam_pos - world_pos);

    RNG rng = make_rng(pid, parameters.frame_index);

    intersector<triangle_data, instancing> inter;
    inter.assume_geometry_type(geometry_type::triangle);

    // TODO: Lights

    float3 radiance   = float3(0.0);
    for (uint sample = 0; sample < parameters.spp; sample++) {
        SurfaceHit hit = current_hit;
        float3 throughput = float3(1.0);
        float3 path_radiance = float3(0.0);
        float3 path_ray_dir = v0;

        for (uint bounce = 0; bounce < parameters.bounce_count; bounce++) {
            path_radiance += throughput * hit.emissive;

            // Directional Light NEE
            float3 l_dir = -light_dir;
            if (dot(hit.n, l_dir) > 0.0) {
                float vis = visibility(hit.pos + hit.n * 0.001, l_dir, 10000.0, scene, ift);
                if (vis > 0.0) {
                    float3 brdf_light = eval_brdf(hit, -path_ray_dir, l_dir);
                    path_radiance += throughput * brdf_light * light_color * vis;
                }
            }

            float3 wi = sample_cosine_hemisphere(hit.n, rng.next_f(), rng.next_f());
            float pdf = saturate(dot(hit.n, wi)) / M_PI_F;
            if (pdf < 1e-5) break;

            float3 brdf_val = eval_brdf(hit, -path_ray_dir, wi);
            throughput *= brdf_val / pdf;

            if (bounce > 1) {
                float p = min(max3(throughput.r, throughput.g, throughput.b), 0.95);
                if (rng.next_f() > p) break;
                throughput /= p;
            }

            ray next;
            next.origin       = hit.pos + hit.n * 0.001;
            next.direction    = wi;
            next.min_distance = 0.001;
            next.max_distance = 10000.0;

            auto result = inter.intersect(next, scene.tlas, 0xFF, ift);
            if (result.type == intersection_type::none) break;

            float3 hit_pos = hit.pos + wi * result.distance;
            hit = fetch_secondary_hit(scene, result.instance_id, result.primitive_id,
                                      result.triangle_barycentric_coord, hit_pos, wi);
            path_ray_dir = wi;
        }

        radiance += path_radiance;
    }
    radiance /= parameters.spp;

    float3 color = current_hit.emissive + radiance;
    float lum = dot(color, float3(0.2126, 0.7152, 0.0722));
    if (lum > 10.0) color *= 10.0 / lum;

    output.write(float4(color, 1.0), pid);
}
