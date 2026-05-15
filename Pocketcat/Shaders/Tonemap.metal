//
//  Tonemap.metal
//  Pocketcat
//
//  Created by Amélie Heinrich on 02/03/2026.
//

#include <metal_stdlib>
#include <simd/simd.h>
using namespace simd;
using namespace metal;

struct parameters {
    float gamma;
    int   mode;
};

float3 agxDefaultContrastApprox(float3 x) {
    float3 x2 = x * x;
    float3 x4 = x2 * x2;
    
    return + 15.5     * x4 * x2
         - 40.14    * x4 * x
         + 31.96    * x4
         - 6.868    * x2 * x
         + 0.4298   * x2
         + 0.1191   * x
         - 0.00232;
}

float3 agx(float3 val) {
    const float3x3 agx_mat = float3x3(
                                      0.842479062253094, 0.0423282422610123, 0.0423756549057051,
                                      0.0784335999999992,  0.878468636469772,  0.0784336,
                                      0.0792237451477643, 0.0791661274605434, 0.879142973793104);
    
    const float min_ev = -12.47393f;
    const float max_ev = 4.026069f;
    
    // Input transform
    val = agx_mat * val;
    
    // Log2 space encoding
    val = clamp(log2(val), min_ev, max_ev);
    val = (val - min_ev) / (max_ev - min_ev);
    
    // Apply sigmoid function approximation
    val = agxDefaultContrastApprox(val);
    
    return val;
}

float3 agxEotf(float3 val) {
    const float3x3 agx_mat_inv = float3x3(
    1.19687900512017, -0.0528968517574562, -0.0529716355144438,
    -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
    -0.0990297440797205, -0.0989611768448433, 1.15107367264116
    );
    
    // Undo input transform
    val = agx_mat_inv * val;
    
    // sRGB IEC 61966-2-1 2.2 Exponent Reference EOTF Display
    //val = pow(val, float3(2.2));
    return val;
}

float3 agxLook(float3 val) {
    // Default
    float3 offset = float3(0.0);
    float3 slope = float3(1.0);
    float3 power = float3(1.0);
    float sat = 1.0;
    
    // ASC CDL
    val = pow(val * slope + offset, power);
    
    const float3 lw = float3(0.2126, 0.7152, 0.0722);
    float luma = dot(val, lw);
    
    return luma + sat * (val - luma);
}

float3 agxTonemap(float3 c) {
    c = agx(c);
    c = agxLook(c);
    c = agxEotf(c);
    return c;
}

float3 acesTonemap(float3 c) {
    // Narkowicz 2015 ACES filmic approximation
    const float a = 2.51f;
    const float b = 0.03f;
    const float cc = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    return saturate((c * (a * c + b)) / (c * (cc * c + d) + e));
}

float3 reinhardTonemap(float3 c) {
    return c / (1.0f + c);
}

float3 uncharted2Partial(float3 x) {
    const float A = 0.15f, B = 0.50f, C = 0.10f, D = 0.20f, E = 0.02f, F = 0.30f;
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

float3 uncharted2Tonemap(float3 c) {
    const float exposure_bias = 2.0f;
    float3 curr = uncharted2Partial(c * exposure_bias);
    float3 white = uncharted2Partial(float3(11.2f));
    return curr / white;
}

struct vs_output {
    float4 position [[position]];
    float2 uv;
};

[[vertex]]
vs_output tonemap_vs(uint vid [[vertex_id]]) {
    float2 positions[3] = {
        float2(-1.0,  1.0),
        float2( 3.0,  1.0),
        float2(-1.0, -3.0)
    };
    float2 uvs[3] = {
        float2(0.0, 0.0),
        float2(2.0, 0.0),
        float2(0.0, 2.0)
    };
    vs_output out;
    out.position = float4(positions[vid], 0.0, 1.0);
    out.uv = uvs[vid];
    return out;
}

[[fragment]]
float4 tonemap_fs(
   vs_output in [[stage_in]],
    texture2d<float> input [[texture(0)]],
    constant parameters& params [[buffer(0)]]
) {
    constexpr sampler s(filter::nearest, address::clamp_to_edge);
    float3 color = input.sample(s, in.uv).xyz;
    float3 mapped;
    switch (params.mode) {
        case 1:  mapped = acesTonemap(color);      break;
        case 2:  mapped = reinhardTonemap(color);  break;
        case 3:  mapped = uncharted2Tonemap(color); break;
        case 4:  mapped = color;                   break; // passthrough
        default: mapped = agxTonemap(color);       break; // AGX (0)
    }
    mapped = pow(mapped, 1.0 / params.gamma);
    return float4(mapped, 1.0);
}
