//
//  AccumulationDenoiser.metal
//  Pocketcat
//
//  Created by Amélie Heinrich on 22/03/2026.
//

#include <metal_stdlib>
using namespace metal;

[[kernel]]
void accumulation_denoiser(texture2d<float>             new_sample   [[texture(0)]],
                           texture2d<float>             motion_vecs  [[texture(1)]],
                           texture2d<float>             prev_accum   [[texture(2)]],
                           texture2d<float, access::write> curr_accum [[texture(3)]],
                           uint2 pid [[thread_position_in_grid]])
{
    uint width  = curr_accum.get_width();
    uint height = curr_accum.get_height();
    if (pid.x >= width || pid.y >= height) return;

    const float max_samples      = 512.0;

    float4 new_col = new_sample.read(pid);
    float2 mv      = motion_vecs.read(pid).rg;

    float2 prev_px = float2(pid) + mv * float2(width, height);
    int2   prev_id = int2(prev_px);

    float4 result;

    bool out_of_bounds = prev_id.x < 0 || prev_id.x >= (int)width ||
                         prev_id.y < 0 || prev_id.y >= (int)height;

    if (out_of_bounds) {
        result = float4(new_col.rgb, 1.0);
    } else {
        float4 old    = prev_accum.read(uint2(prev_id));
        float  count  = old.a;
        float  motion = length(mv);

        if (count < 1.0 || motion != 0.0f) {
            result = float4(new_col.rgb, 1.0);
        } else {
            float  new_count = min(count + 1.0, max_samples);
            float3 new_color = old.rgb + (new_col.rgb - old.rgb) / new_count;
            result = float4(new_color, new_count);
        }
    }

    curr_accum.write(result, pid);
}
