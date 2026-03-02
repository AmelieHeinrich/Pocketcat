//
//  TextureCompressor.h
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 02/03/2026.
//

#pragma once

#include <AppleTextureEncoder.h>
#include <AppleTextureConverter.h>
#include <string>

// All fields are uint32_t so sizeof(TextureHeader) == 16 with no padding.
// MTLPixelFormat values fit in 32 bits; using NSUInteger/MTLPixelFormat directly
// would add 4 bytes of trailing padding and misalign the payload.
struct TextureHeader
{
    uint32_t Format;
    uint32_t Width;
    uint32_t Height;
    uint32_t MipLevels;
    
    ATC_Compressor test;
};

void CompressTexture(const std::string& source, const std::string& out);
