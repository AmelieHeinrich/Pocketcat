//
//  ResetICB.metal
//  Pocketcat
//
//  Created by Amélie Heinrich on 08/03/2026.
//

#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
using namespace simd;

struct ICBWrapper {
    command_buffer CommandBuffer;
};

[[kernel]]
void reset_icb(device ICBWrapper& icb [[buffer(0)]],
               constant uint& commandCount [[buffer(1)]],
               uint threadID [[thread_position_in_grid]]) {
    if (threadID >= commandCount) return;
    uint commandIndex = threadID;

    render_command command(icb.CommandBuffer, commandIndex);
    command.reset();
}
