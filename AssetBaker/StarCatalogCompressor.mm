//
//  StarCatalogCompressor.mm
//  AssetBaker
//
//  Created by Amélie Heinrich on 15/05/2026.
//

#include "StarCatalogCompressor.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::string getColumn(const std::vector<std::string>& cols, int idx)
{
    if (idx < 0 || idx >= (int)cols.size()) return "";
    return cols[idx];
}

static std::vector<std::string> splitCSV(const std::string& line)
{
    std::vector<std::string> result;
    bool inQuotes = false;
    std::string token;
    for (char c : line) {
        if (c == '"') {
            inQuotes = !inQuotes;
        } else if (c == ',' && !inQuotes) {
            result.push_back(token);
            token.clear();
        } else {
            token += c;
        }
    }
    result.push_back(token);
    return result;
}

void CompressStarCatalog(const std::string& source, const std::string& out)
{
    std::ifstream in(source);
    if (!in) {
        fprintf(stderr, "[StarCatalogCompressor] Cannot open: %s\n", source.c_str());
        return;
    }

    // HYG v42 columns (0-based):
    //  7  = ra (hours)        23 = rarad
    //  8  = dec (degrees)     24 = decrad
    // 13  = mag
    // 16  = ci (B-V)
    constexpr int COL_RARAD  = 23;
    constexpr int COL_DECRAD = 24;
    constexpr int COL_MAG    = 13;
    constexpr int COL_CI     = 16;
    constexpr float MAG_LIMIT = 7.5f;

    std::vector<StarEntry> stars;
    stars.reserve(120000);

    std::string line;
    std::getline(in, line); // skip header

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        auto cols = splitCSV(line);

        const std::string& magStr = getColumn(cols, COL_MAG);
        if (magStr.empty()) continue;
        float mag = std::stof(magStr);
        if (mag >= MAG_LIMIT) continue;

        const std::string& raStr  = getColumn(cols, COL_RARAD);
        const std::string& decStr = getColumn(cols, COL_DECRAD);
        const std::string& ciStr  = getColumn(cols, COL_CI);

        float ra  = raStr.empty()  ? 0.0f : std::stof(raStr);
        float dec = decStr.empty() ? 0.0f : std::stof(decStr);
        float bv  = ciStr.empty()  ? 0.0f : std::stof(ciStr);

        stars.push_back({ ra, dec, mag, bv });
    }

    std::ofstream file(out, std::ios::binary);
    if (!file) {
        fprintf(stderr, "[StarCatalogCompressor] Cannot write: %s\n", out.c_str());
        return;
    }

    StarCatalogHeader header;
    header.count = static_cast<uint32_t>(stars.size());
    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    file.write(reinterpret_cast<const char*>(stars.data()),
               static_cast<std::streamsize>(stars.size() * sizeof(StarEntry)));

    printf("[StarCatalogCompressor] Wrote %u stars to %s\n", header.count, out.c_str());
}
