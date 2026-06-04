#include <metal_stdlib>
using namespace metal;

struct ColorCorrectionParams {
    float temperature;  // kelvin offset, ± range; positive = warm, negative = cool
    float tint;         // green-magenta offset
    float exposure;     // EV stops
    float contrast;     // pivot-based contrast (1.0 = neutral)
    float brightness;   // additive lift
    float saturation;   // 1.0 = neutral
};

// Approximate white balance shift in linear sRGB from a kelvin offset.
// We map temperature to a red/blue tilt and tint to a green tilt.
float3 applyWhiteBalance(float3 c, float temp, float tint) {
    float r = c.r + temp * 0.0001;
    float g = c.g + tint  * 0.0001;
    float b = c.b - temp * 0.0001;
    return float3(r, g, b);
}

kernel void color_correction(
    texture2d<float, access::read_write> hdr [[texture(0)]],
    constant ColorCorrectionParams& p        [[buffer(0)]],
    uint2 tid                                [[thread_position_in_grid]]
)
{
    if (tid.x >= hdr.get_width() || tid.y >= hdr.get_height()) return;

    float3 c = hdr.read(tid).rgb;

    // Exposure (applied in linear light)
    c *= pow(2.0, p.exposure);

    // White balance
    c = applyWhiteBalance(c, p.temperature, p.tint);

    // Contrast around a mid-grey pivot of 0.18
    const float pivot = 0.18;
    c = (c - pivot) * p.contrast + pivot;

    // Brightness (additive lift)
    c += p.brightness;

    // Saturation via luminance
    const float3 luma_weights = float3(0.2126, 0.7152, 0.0722);
    float lum = dot(c, luma_weights);
    c = mix(float3(lum), c, p.saturation);

    hdr.write(float4(c, 1.0), tid);
}
