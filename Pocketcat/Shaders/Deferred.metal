//
//  Deferred.metal
//  Pocketcat
//
//  Created by Amélie Heinrich on 21/03/2026.
//

#include "Common/Bindless.h"
#include "Common/Math.h"
#include "Common/PBR.h"

[[kernel]]
void deferred_kernel(const device scene_data& scene [[buffer(0)]],
                     intersection_function_table<triangle_data, instancing> ift [[buffer(1)]],
                     texture2d<float> depth_texture [[texture(0)]],
                     texture2d<float> albedo_texture [[texture(1)]],
                     texture2d<float> normal_texture [[texture(2)]],
                     texture2d<float> orm_texture [[texture(3)]],
                     texture2d<float> emissive_texture [[texture(4)]],
                     texture2d<float, access::write> output_texture [[texture(5)]],
                     uint2 gtid [[thread_position_in_grid]])
{
    if (gtid.x >= output_texture.get_width() || gtid.y >= output_texture.get_height()) {
        return;
    }

    const float3 light_dir = scene.sun.direction_and_radius.xyz;
    const float3 light_color = scene.sun.color_and_intensity.xyz * scene.sun.color_and_intensity.w;

    float depth = depth_texture.read(gtid).r;
    if (depth == 1.0) {
        output_texture.write(0, gtid);
        return;
    }

    float2 uv = (float2(gtid) + 0.5) / float2(output_texture.get_width(), output_texture.get_height());
    float2 ndc = uv * 2.0 - 1.0;
    ndc.y = -ndc.y;

    float4 clip_pos = float4(ndc, depth, 1.0);
    float4 world_pos_h = scene.camera.inverse_view_projection * clip_pos;
    float3 world_pos = world_pos_h.xyz / world_pos_h.w;

    float3 albedo = albedo_texture.read(gtid).rgb;

    // Sample and unpack normal
    float3 normal = normal_texture.read(gtid).rgb;

    float2 rm = orm_texture.read(gtid).rg;
    float roughness = max(rm.r, 0.04);
    float metallic = rm.g;

    float3 emissive = emissive_texture.read(gtid).rgb;

    float3 V = normalize(scene.camera.position_and_near.xyz - world_pos);
    float3 L = normalize(-light_dir);
    float3 H = normalize(V + L);

    float NdotL = saturate(dot(normal, L));
    float NdotV = saturate(dot(normal, V));
    float NdotH = saturate(dot(normal, H));
    float VdotH = saturate(dot(V, H));

    float3 F0 = float3(0.04);
    F0 = mix(F0, albedo, metallic);

    float D = d_ggx(NdotH, roughness);
    float G = g_smith(NdotV, NdotL, roughness);
    float3 F = f_schlick(VdotH, F0);

    float3 numerator = D * G * F;
    float denominator = 4.0 * max(NdotV, 0.001) * max(NdotL, 0.001);
    float3 specular = numerator / max(denominator, 0.001);

    float3 kS = F;
    float3 kD = float3(1.0) - kS;
    kD *= 1.0 - metallic;
    
    ray shadow_ray;
    shadow_ray.origin  = world_pos + normal * 0.01;
    shadow_ray.direction = -light_dir;
    shadow_ray.min_distance = 0.001;
    shadow_ray.max_distance = 10000;

    intersector<triangle_data, instancing> shadow_inter;
    shadow_inter.assume_geometry_type(geometry_type::triangle);
    shadow_inter.accept_any_intersection(true);

    typename intersector<triangle_data, instancing>::result_type shadow_result;
    shadow_result = shadow_inter.intersect(shadow_ray, scene.tlas, 0xFF, ift);
    float shadow = (shadow_result.type == intersection_type::none) ? 1.0 : 0.0;

    float3 diffuse = (kD * albedo / M_PI_F);
    float3 Lo = (diffuse + specular) * light_color * NdotL * shadow;

    float3 color = Lo + emissive;
    output_texture.write(float4(color, 1.0), gtid);
}
