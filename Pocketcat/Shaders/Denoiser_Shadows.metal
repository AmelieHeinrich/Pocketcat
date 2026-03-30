//
//  Denoiser_Shadows.metal
//  Pocketcat
//
//  Created by Amélie Heinrich on 25/03/2026.
//

#include "Common/Bindless.h"

struct temporal_input
{
    texture2d<float> shadow_mask;
    texture2d<float> motion_vectors;
    texture2d<float> current_normals;
    texture2d<float> previous_normals;
    texture2d<float, access::read_write> filtered;
    texture2d<float> previous_filtered;
    texture2d<float, access::read_write> moments;
    texture2d<float> previous_moments;
    texture2d<float, access::read_write> history;
    float depth_threshold;
    float normal_threshold;
};

// Weight functions used for à-trous
float normal_edge_stopping_weight(float3 center_normal, float3 sample_normal, float power)
{
    return pow(clamp(dot(center_normal, sample_normal), 0.0f, 1.0f), power);
}

float depth_edge_stopping_weight(float center_depth, float sample_depth, float phi)
{
    return exp(-abs(center_depth - sample_depth) / phi);
}

float luma_edge_stopping_weight(float center_luma, float sample_luma, float phi)
{
    return abs(center_luma - sample_luma) / phi;
}

// Reprojection checks
// Checks if the reprojected pixel coordinates fall outside the screen boundaries
bool out_of_frame_disocclusion_check(int2 coord, int2 size)
{
    if (coord.x < 0 || coord.y < 0 || coord.x >= size.x || coord.y >= size.y)
        return true;
    return false;
}

// Computes a continuous weight based on normal and depth differences to detect disocclusions
float normal_depth_disocclusion_check(float3 normal, float3 history_normal, float linear_depth, float history_linear_depth)
{
    return 1.0 * exp(-abs(1.0 - max(0.0, dot(normal, history_normal))) * 1.4) * exp(-abs(history_linear_depth - linear_depth) / max(linear_depth, 1e-6) * 1.0);
}

// Validates if a historical sample can be reused based on screen bounds and normal/depth similarity
bool is_reprojection_valid(int2 coord, int2 size, float current_linear_depth, float history_linear_depth, float3 current_normal, float3 history_normal)
{
    if (out_of_frame_disocclusion_check(coord, size)) return false;
    if (normal_depth_disocclusion_check(current_normal, history_normal, current_linear_depth, history_linear_depth) < 0.9) return false;
    return true;
}

// 5x5 neighborhood mean and variance calculation. If a pixel successfully fetches history, it clips that historical shadow value to the local valid bounds of the current frame to kill ghosting and smearing
float clip_aabb(float aabb_min, float aabb_max, float history_sample)
{
    float aabb_center = 0.5f * (aabb_max + aabb_min);
    float extent_clip = 0.5f * (aabb_max - aabb_min) + 0.001f;

    float color_vector = history_sample - aabb_center;
    float color_vector_clip = color_vector / extent_clip;
    color_vector_clip = abs(color_vector_clip);

    if (color_vector_clip > 1.0)
        return aabb_center + color_vector / color_vector_clip;
    else
        return history_sample;
}

bool load_prev_data(const device temporal_input& input, int2 frag_coord, float2 history_coord, float depth, float3 current_normal, thread float& history_shadow, thread float2& history_moments, thread float& history_length)
{
    int2 size = int2(input.shadow_mask.get_width(), input.shadow_mask.get_height());
    int2 ipos_prev = int2(history_coord + 0.5f);

    bool v[4];
    float2 posPrev = history_coord;
    int2 offset[4] = { int2(0, 0), int2(1, 0), int2(0, 1), int2(1, 1) };

    // Check all 4 taps of the bilinear filter for validity
    bool valid = false;
    for (int sampleIdx = 0; sampleIdx < 4; sampleIdx++) {
        int2 loc = int2(posPrev) + offset[sampleIdx];
        float history_linear_depth = input.motion_vectors.read(uint2(clamp(loc, int2(0), size - 1))).z;
        float3 history_normal = input.previous_normals.read(uint2(clamp(loc, int2(0), size - 1))).rgb;
        v[sampleIdx] = is_reprojection_valid(ipos_prev, size, depth, history_linear_depth, current_normal, history_normal);
        valid = valid || v[sampleIdx];
    }

    history_shadow = 0.0;
    history_moments = 0.0;

    if (valid) {
        float sumw = 0;
        float x = fract(posPrev.x);
        float y = fract(posPrev.y);
        float w[4] = { (1 - x) * (1 - y), x * (1 - y), (1 - x) * y, x * y };

        for (int sampleIdx = 0; sampleIdx < 4; sampleIdx++) {
            int2 loc = int2(posPrev) + offset[sampleIdx];
            uint2 uloc = uint2(clamp(loc, int2(0), size - 1));
            if (v[sampleIdx]) {
                history_shadow += w[sampleIdx] * input.previous_filtered.read(uloc).r;
                history_moments += w[sampleIdx] * input.previous_moments.read(uloc).rg;
                sumw += w[sampleIdx];
            }
        }

        // Redistribute weights in case not all taps were used
        valid = (sumw >= 0.01);
        history_shadow = valid ? history_shadow / sumw : 0.0f;
        history_moments = valid ? history_moments / sumw : 0.0f;
    }

    // If bilinear fails, perform a 3x3 cross-bilateral filter fallback to find a valid sample nearby
    if (!valid) {
        float cnt = 0.0;
        for (int yy = -1; yy <= 1; yy++) {
            for (int xx = -1; xx <= 1; xx++) {
                int2 p = ipos_prev + int2(xx, yy);
                uint2 up = uint2(clamp(p, int2(0), size - 1));
                float history_linear_depth = input.motion_vectors.read(up).z;
                float3 history_normal = input.previous_normals.read(up).rgb;

                if (is_reprojection_valid(ipos_prev, size, depth, history_linear_depth, current_normal, history_normal)) {
                    history_shadow += input.previous_filtered.read(up).r;
                    history_moments += input.previous_moments.read(up).rg;
                    cnt += 1.0;
                }
            }
        }
        if (cnt > 0) {
            valid = true;
            history_shadow /= cnt;
            history_moments /= cnt;
        }
    }

    if (valid) {
        history_length = input.history.read(uint2(clamp(ipos_prev, int2(0), size - 1))).r;
    } else {
        history_shadow = 0.0f;
        history_moments = 0.0f;
    }

    return valid;
}

[[kernel]]
void denoise_shadows_temporal(const device temporal_input& input [[buffer(0)]],
                              uint2 gtid [[thread_position_in_grid]])
{
    uint width = input.shadow_mask.get_width();
    uint height = input.shadow_mask.get_height();
    if (gtid.x >= width || gtid.y >= height)
        return;

    int2 current_coord = int2(gtid);
    float depth = input.motion_vectors.read(gtid).z;

    if (depth == 1.0f) {
        input.filtered.write(0.0f, gtid);
        input.moments.write(0.0f, gtid);
        input.history.write(0.0f, gtid);
        return;
    }

    float shadow = input.shadow_mask.read(gtid).r;
    float3 normal = input.current_normals.read(gtid).rgb;
    float2 velocity = input.motion_vectors.read(gtid).rg;

    float2 history_coord = float2(current_coord) + (velocity * float2(width, height));
    float history_shadow = 0.0f;
    float2 history_moments = 0.0f;
    float history_length = 0.0f;

    bool success = load_prev_data(input, current_coord, history_coord, depth, normal, history_shadow, history_moments, history_length);

    // Cap history length to 32 frames otherwise there will be tons of ghosting
    history_length = min(32.0f, success ? history_length + 1.0f : 1.0f);

    if (success) {
        // Compute spatial mean and standard deviation of the current frame in a 3x3 neighborhood
        float m1 = 0.0f;
        float m2 = 0.0f;
        float weight = 0.0f;
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                int2 sample_coord = clamp(current_coord + int2(dx, dy), int2(0), int2(width - 1, height - 1));
                float sample_color = input.shadow_mask.read(uint2(sample_coord)).r;
                m1 += sample_color;
                m2 += sample_color * sample_color;
                weight += 1.0f;
            }
        }
        float mean = m1 / weight;
        float variance = (m2 / weight) - (mean * mean);
        float std_dev = sqrt(max(variance, 0.0f));

        float radiance_min = mean - (1.0f * std_dev);
        float radiance_max = mean + (1.0f * std_dev);

        // Clip the history shadow to the current frame's valid local variance range (AABB)
        history_shadow = clip_aabb(radiance_min, radiance_max, history_shadow);
    }

    float alpha = success ? max(0.05f, 1.0f / history_length) : 1.0f;

    // Compute first two moments of luminance for variance estimation and temporally integrate
    float2 moments = float2(shadow, shadow * shadow);
    moments = mix(history_moments, moments, alpha);

    // Write out temporal moments and variance (variance = E[x^2] - E[x]^2)
    input.moments.write(float4(moments.x, moments.y, max(0.0f, moments.y - moments.x * moments.x), 1.0), gtid);
    input.history.write(float4(history_length, 0.0f, 0.0f, 0.0f), gtid);

    // Temporally integrate the shadow mask
    float accumulated_shadow = mix(history_shadow, shadow, alpha);
    input.filtered.write(accumulated_shadow, gtid);
}

// Variance estimation

struct variance_estimation_input
{
    texture2d<float> input_shadow;
    texture2d<float> normals;
    texture2d<float> motion_vectors;
    texture2d<float> moments;
    texture2d<float> history;
    texture2d<float, access::read_write> output_shadow;
    float phi_color;
    float phi_normal;
};

[[kernel]]
void denoise_shadows_variance_estimation(const device variance_estimation_input& input [[buffer(0)]],
                                         uint2 gtid [[thread_position_in_grid]])
{
    int2 size = int2(input.input_shadow.get_width(), input.input_shadow.get_height());
    if (int(gtid.x) >= size.x || int(gtid.y) >= size.y)
        return;

    float center_color = input.input_shadow.read(gtid).r;
    float3 center_normal = input.normals.read(gtid).rgb;
    float center_depth = input.motion_vectors.read(gtid).z;

    if (center_depth == 1.0f) {
        input.output_shadow.write(float4(center_color, 0.0f, 0.0f, 0.0f), gtid);
        return;
    }

    float history_length = input.history.read(gtid).r;
    float4 out_color = float4(center_color, 0.0f, 0.0f, 0.0f);

    if (history_length < 4.0f) {
        float sum_w = 1.0f;
        float sum_color = center_color;
        float2 sum_moments = input.moments.read(gtid).rg;

        for (int yy = -3; yy <= 3; yy++) {
            for (int xx = -3; xx <= 3; xx++) {
                int2 p = int2(gtid) + int2(xx, yy);
                bool inside = p.x >= 0 && p.y >= 0 && p.x < size.x && p.y < size.y;

                if (inside) {
                    float sample_color = input.input_shadow.read(uint2(p)).r;
                    float2 sample_moments = input.moments.read(uint2(p)).rg;
                    float3 sample_normal = input.normals.read(uint2(p)).rgb;
                    float sample_depth = input.motion_vectors.read(uint2(p)).z;

                    float wNormal = normal_edge_stopping_weight(center_normal, sample_normal, input.phi_normal);
                    float wZ = abs(center_depth - sample_depth) / 1.0f;
                    float wL = abs(center_color - sample_color) / input.phi_color;

                    float w = exp(0.0f - max(wL, 0.0f) - max(wZ, 0.0f)) * wNormal;

                    sum_w += w;
                    sum_color += sample_color * w;
                    sum_moments += sample_moments * w;
                }
            }
        }

        sum_w = max(sum_w, 1e-6f);
        sum_color /= sum_w;
        sum_moments /= sum_w;

        float variance = sum_moments.g - sum_moments.r * sum_moments.r;
        variance *= 4.0f / history_length;

        out_color = float4(sum_color, variance, 0.0f, 0.0f);
    } else {
        float variance = input.moments.read(gtid).b;
        out_color = float4(center_color, variance, 0.0f, 0.0f);
    }

    input.output_shadow.write(out_color, gtid);
}

// À-trous

float compute_variance_center(int2 ipos, texture2d<float> input_shadow, int2 size)
{
    float sum = 0.0f;
    const float blur_kernel[2][2] = {
        { 1.0f / 4.0f, 1.0f / 8.0f },
        { 1.0f / 8.0f, 1.0f / 16.0f }
    };

    for (int yy = -1; yy <= 1; yy++) {
        for (int xx = -1; xx <= 1; xx++) {
            int2 p = clamp(ipos + int2(xx, yy), int2(0), size - 1);
            float k = blur_kernel[abs(xx)][abs(yy)];
            sum += input_shadow.read(uint2(p)).g * k;
        }
    }
    return sum;
}

struct atrous_input
{
    texture2d<float> input_shadow;
    texture2d<float> motion_vectors;
    texture2d<float> normals;
    texture2d<float, access::read_write> output_shadow;
    int step_size;
    float phi_color;
    float phi_normal;
};

[[kernel]]
void denoise_shadows_atrous(const device atrous_input& input [[buffer(0)]],
                            uint2 gtid [[thread_position_in_grid]])
{
    int2 size = int2(input.input_shadow.get_width(), input.input_shadow.get_height());
    if (int(gtid.x) >= size.x || int(gtid.y) >= size.y)
        return;

    int2 ipos = int2(gtid);
    const float eps_variance = 1e-10f;
    const float kernel_weights[3] = { 1.0f, 2.0f / 3.0f, 1.0f / 6.0f };

    float2 center_color_var = input.input_shadow.read(gtid).rg;
    float center_color = center_color_var.r;
    float3 center_normal = input.normals.read(gtid).rgb;
    float center_depth = input.motion_vectors.read(gtid).z;

    if (center_depth == 1.0f) {
        input.output_shadow.write(float4(0.0f), gtid);
        return;
    }

    float var = compute_variance_center(ipos, input.input_shadow, size);
    float phi_color = input.phi_color * sqrt(max(0.0f, eps_variance + var));

    float sum_w = 1.0f;
    float2 sum_color_var = center_color_var;

    for (int yy = -1; yy <= 1; yy++) {
        for (int xx = -1; xx <= 1; xx++) {
            int2 p = ipos + int2(xx, yy) * input.step_size;
            bool inside = p.x >= 0 && p.y >= 0 && p.x < size.x && p.y < size.y;

            if (inside && (xx != 0 || yy != 0)) {
                float2 sample_color_var = input.input_shadow.read(uint2(p)).rg;
                float3 sample_normal = input.normals.read(uint2(p)).rgb;
                float sample_depth = input.motion_vectors.read(uint2(p)).z;

                float wNormal = normal_edge_stopping_weight(center_normal, sample_normal, input.phi_normal);
                float wZ = abs(center_depth - sample_depth) / float(input.step_size);
                float wL = abs(center_color - sample_color_var.r) / phi_color;

                float w = exp(0.0f - max(wL, 0.0f) - max(wZ, 0.0f)) * wNormal;
                float w_kern = w * kernel_weights[abs(xx)] * kernel_weights[abs(yy)];

                sum_w += w_kern;
                sum_color_var.r += sample_color_var.r * w_kern;
                sum_color_var.g += sample_color_var.g * w_kern * w_kern;
            }
        }
    }

    float2 out_color_var = float2(sum_color_var.r / sum_w, sum_color_var.g / (sum_w * sum_w));
    input.output_shadow.write(float4(out_color_var.r, out_color_var.g, 0.0f, 0.0f), gtid);
}
