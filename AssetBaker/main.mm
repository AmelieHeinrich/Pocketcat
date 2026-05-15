//
//  main.mm
//  AssetBaker
//
//  Created by Amélie Heinrich on 02/03/2026.
//

// Iterates over Assets/SourceAssets and bakes every .gltf into Assets/BakedAssets,
// preserving the directory structure and changing the extension to .bin.
// Run from the project root, or pass the project root as argv[1].

#include "MeshCompressor.h"
#include "StarCatalogCompressor.h"

#include <cstdio>
#include <filesystem>
#include <string>

// Forward-declare to avoid pulling in AppleTextureConverter.h via TextureCompressor.h
void CompressTexture(const std::string& source, const std::string& out);

namespace fs = std::filesystem;

int main(int argc, const char* argv[])
{
    fs::path root = (argc > 1) ? fs::path(argv[1]) : fs::current_path();

    fs::path sourceDir = root / "Assets" / "SourceAssets";
    fs::path bakedDir  = root / "Assets" / "BakedAssets";

    if (!fs::exists(sourceDir)) {
        fprintf(stderr, "[AssetBaker] Source directory not found: %s\n", sourceDir.c_str());
        return 1;
    }

    int baked  = 0;
    int failed = 0;

    for (const fs::directory_entry& entry : fs::recursive_directory_iterator(sourceDir)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".gltf")
            continue;

        fs::path rel    = fs::relative(entry.path(), sourceDir);
        fs::path outPath = (bakedDir / rel).replace_extension(".bin");

        fs::create_directories(outPath.parent_path());

        printf("[AssetBaker] Baking: %s\n", rel.c_str());
        CompressMesh(entry.path().string(), outPath.string());
        baked++;
    }

    printf("[AssetBaker] Done. Baked %d mesh(es).\n", baked + failed);

    // --- Astronomy assets ---
    fs::path astroSrc   = root / "Assets" / "Astronomy";
    fs::path astroBaked = root / "Assets" / "BakedAssets" / "Astronomy";
    fs::create_directories(astroBaked);

    // HYG CSV → binary .star
    fs::path csvPath  = astroSrc  / "hyg_v42.csv";
    fs::path starPath = astroBaked / "hyg_v42.star";
    if (fs::exists(csvPath)) {
        printf("[AssetBaker] Baking star catalog: hyg_v42.csv\n");
        CompressStarCatalog(csvPath.string(), starPath.string());
    }

    // PNG textures → ASTC .tex
    for (const fs::directory_entry& entry : fs::directory_iterator(astroSrc)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".png")
            continue;
        fs::path outTex = (astroBaked / entry.path().filename()).replace_extension(".tex");
        printf("[AssetBaker] Compressing texture: %s\n", entry.path().filename().c_str());
        CompressTexture(entry.path().string(), outTex.string());
    }

    return 0;
}
