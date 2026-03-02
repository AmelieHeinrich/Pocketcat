//
//  TextureCompressor.h
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 02/03/2026.
//

#pragma once

#include <AppleTextureEncoder.h>
#include <string>

#include <Metal/Metal.h>

struct TextureHeader
{
    MTLPixelFormat Format;
    uint32_t Width;
    uint32_t Height;
    uint32_t MipLevels;
};

void CompressTexture(const std::string& source, const std::string& out);
