//
//  StarCatalogCompressor.h
//  AssetBaker
//
//  Created by Amélie Heinrich on 15/05/2026.
//

#pragma once

#include <string>
#include <cstdint>

// Binary .star format:
//   StarCatalogHeader (4 bytes)
//   StarEntry[count]  (16 bytes each)
struct StarCatalogHeader {
    uint32_t count;
};

struct StarEntry {
    float ra;           // radians
    float dec;          // radians
    float mag;          // apparent magnitude
    float bv;           // B-V color index
};

// Parses hyg_v42.csv at `source`, filters mag < 7.5, writes binary to `out`.
void CompressStarCatalog(const std::string& source, const std::string& out);
