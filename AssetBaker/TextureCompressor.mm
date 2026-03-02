//
//  TextureCompressor.cpp
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 02/03/2026.
//

#include "TextureCompressor.h"

#define STB_IMAGE_IMPLEMENTATION
#include "ThirdParty/stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "ThirdParty/stb_image_resize2.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

static uint32_t NumMipLevels(uint32_t width, uint32_t height)
{
    return static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1;
}

static size_t ASTCMipSize(uint32_t width, uint32_t height)
{
    uint32_t blockW = (width  + 3) / 4;
    uint32_t blockH = (height + 3) / 4;
    return static_cast<size_t>(blockW) * blockH * 16;
}

void CompressTexture(const std::string& source, const std::string& out)
{
    // Load source image as RGBA8
    int width, height, channels;
    stbi_uc* pixels = stbi_load(source.c_str(), &width, &height, &channels, 4);
    if (!pixels) {
        fprintf(stderr, "[TextureCompressor] Failed to load: %s\n", source.c_str());
        return;
    }

    uint32_t mipLevels = NumMipLevels(static_cast<uint32_t>(width), static_cast<uint32_t>(height));

    // Create ASTC 4x4 encoder: RGBA8 texels -> ASTC 4x4 LDR blocks
    at_encoder_t encoder = at_encoder_create(
        at_texel_format_rgba8_unorm,
        at_alpha_not_premultiplied,
        at_block_format_astc_4x4_ldr,
        at_alpha_not_premultiplied,
        nullptr
    );
    if (!encoder) {
        fprintf(stderr, "[TextureCompressor] Failed to create at_encoder_t\n");
        stbi_image_free(pixels);
        return;
    }

    // Pre-calculate per-mip dimensions and offsets into the compressed buffer
    std::vector<uint32_t> mipWidths(mipLevels);
    std::vector<uint32_t> mipHeights(mipLevels);
    std::vector<size_t>   mipOffsets(mipLevels);
    {
        uint32_t w = static_cast<uint32_t>(width);
        uint32_t h = static_cast<uint32_t>(height);
        size_t   offset = 0;
        for (uint32_t i = 0; i < mipLevels; i++) {
            mipWidths[i]  = w;
            mipHeights[i] = h;
            mipOffsets[i] = offset;
            offset += ASTCMipSize(w, h);
            w = std::max(1u, w / 2);
            h = std::max(1u, h / 2);
        }
    }
    size_t totalCompressedSize = mipOffsets.back() + ASTCMipSize(mipWidths.back(), mipHeights.back());

    // Allocate a 16-byte aligned buffer — required by at_block_buffer_t
    void* rawBuffer = nullptr;
    if (posix_memalign(&rawBuffer, 16, totalCompressedSize) != 0) {
        fprintf(stderr, "[TextureCompressor] Failed to allocate compressed buffer\n");
        stbi_image_free(pixels);
        return;
    }
    uint8_t* compressedData = static_cast<uint8_t*>(rawBuffer);

    // Compress each mip level
    std::vector<uint8_t> mipPixels(static_cast<size_t>(width) * height * 4);
    memcpy(mipPixels.data(), pixels, mipPixels.size());
    stbi_image_free(pixels);

    for (uint32_t mip = 0; mip < mipLevels; mip++) {
        uint32_t mipW = mipWidths[mip];
        uint32_t mipH = mipHeights[mip];

        at_texel_region_t src = {};
        src.texels     = mipPixels.data();
        src.validSize  = { mipW, mipH, 1 };
        src.rowBytes   = mipW * 4;
        src.sliceBytes = mipW * mipH * 4;

        uint32_t blockW = (mipW + 3) / 4;
        at_block_buffer_t dest = {};
        dest.blocks    = compressedData + mipOffsets[mip];
        dest.rowBytes  = blockW * 16;
        dest.sliceBytes = ASTCMipSize(mipW, mipH);

        float err = at_encoder_compress_texels(encoder, &src, &dest, 1.0f / (1 << 10), at_flags_default);
        if (err < 0.0f)
            fprintf(stderr, "[TextureCompressor] Mip %u compression failed (err=%f)\n", mip, err);

        // Downsample to produce the next mip
        if (mip + 1 < mipLevels) {
            uint32_t nextW = mipWidths[mip + 1];
            uint32_t nextH = mipHeights[mip + 1];
            std::vector<uint8_t> nextPixels(static_cast<size_t>(nextW) * nextH * 4);
            stbir_resize_uint8_srgb(
                mipPixels.data(), static_cast<int>(mipW), static_cast<int>(mipH), static_cast<int>(mipW * 4),
                nextPixels.data(), static_cast<int>(nextW), static_cast<int>(nextH), static_cast<int>(nextW * 4),
                STBIR_RGBA
            );
            mipPixels = std::move(nextPixels);
        }
    }

    // Write output: TextureHeader followed by tightly packed mip data
    TextureHeader header = {};
    header.Format    = MTLPixelFormatASTC_4x4_LDR;
    header.Width     = static_cast<uint32_t>(width);
    header.Height    = static_cast<uint32_t>(height);
    header.MipLevels = mipLevels;

    std::ofstream file(out, std::ios::binary);
    if (!file) {
        fprintf(stderr, "[TextureCompressor] Failed to open output: %s\n", out.c_str());
        free(rawBuffer);
        return;
    }
    file.write(reinterpret_cast<const char*>(&header), sizeof(TextureHeader));
    file.write(reinterpret_cast<const char*>(compressedData), static_cast<std::streamsize>(totalCompressedSize));

    free(rawBuffer);
}
