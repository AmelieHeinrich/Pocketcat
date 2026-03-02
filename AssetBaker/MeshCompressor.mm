//
//  MeshCompressor.mm
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 02/03/2026.
//

#include "MeshCompressor.h"
#include "TextureCompressor.h"

#define CGLTF_IMPLEMENTATION
#include "ThirdParty/cgltf.h"

#include "ThirdParty/meshoptimizer.h"
#include "ThirdParty/mikktspace.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ---- MikktSpace -------------------------------------------------------------
// MikktSpace requires unindexed geometry: each face indexes sequentially into
// a flat vertex array (face i uses verts[i*3 + 0..2]). We expand the indexed
// mesh before calling genTangSpaceDefault, then re-index with meshopt afterward.

struct MikktMesh
{
    MeshVertex* verts;
    size_t      triCount;
};

static int mkt_getNumFaces(const SMikkTSpaceContext* ctx)
{
    return static_cast<int>(static_cast<MikktMesh*>(ctx->m_pUserData)->triCount);
}

static int mkt_getNumVertsOfFace(const SMikkTSpaceContext*, int)
{
    return 3;
}

static void mkt_getPosition(const SMikkTSpaceContext* ctx, float out[], int face, int vert)
{
    const float* p = static_cast<MikktMesh*>(ctx->m_pUserData)->verts[face * 3 + vert].Position;
    out[0] = p[0]; out[1] = p[1]; out[2] = p[2];
}

static void mkt_getNormal(const SMikkTSpaceContext* ctx, float out[], int face, int vert)
{
    const float* n = static_cast<MikktMesh*>(ctx->m_pUserData)->verts[face * 3 + vert].Normal;
    out[0] = n[0]; out[1] = n[1]; out[2] = n[2];
}

static void mkt_getTexCoord(const SMikkTSpaceContext* ctx, float out[], int face, int vert)
{
    const float* uv = static_cast<MikktMesh*>(ctx->m_pUserData)->verts[face * 3 + vert].UV;
    out[0] = uv[0]; out[1] = uv[1];
}

static void mkt_setTSpaceBasic(const SMikkTSpaceContext* ctx, const float tangent[], float sign, int face, int vert)
{
    float* t = static_cast<MikktMesh*>(ctx->m_pUserData)->verts[face * 3 + vert].Tangent;
    t[0] = tangent[0]; t[1] = tangent[1]; t[2] = tangent[2]; t[3] = sign;
}

static void GenerateTangents(std::vector<MeshVertex>& expanded)
{
    MikktMesh mesh = { expanded.data(), expanded.size() / 3 };

    SMikkTSpaceInterface iface = {};
    iface.m_getNumFaces          = mkt_getNumFaces;
    iface.m_getNumVerticesOfFace = mkt_getNumVertsOfFace;
    iface.m_getPosition          = mkt_getPosition;
    iface.m_getNormal            = mkt_getNormal;
    iface.m_getTexCoord          = mkt_getTexCoord;
    iface.m_setTSpaceBasic       = mkt_setTSpaceBasic;

    SMikkTSpaceContext ctx = { &iface, &mesh };
    genTangSpaceDefault(&ctx);
}

// ---- Helpers ----------------------------------------------------------------

static std::string OutputTexturePath(const std::string& outDir, const char* uri)
{
    return outDir + "/" + fs::path(uri).stem().string() + ".tex";
}

static void ProcessTexture(const std::string& gltfDir, const std::string& outDir,
                           const char* uri, char* outPath, size_t outPathSize)
{
    if (!uri || uri[0] == '\0') {
        outPath[0] = '\0';
        return;
    }
    std::string src = gltfDir + "/" + uri;
    std::string dst = OutputTexturePath(outDir, uri);
    CompressTexture(src, dst);
    snprintf(outPath, outPathSize, "%s", dst.c_str());
}

// ---- CompressMesh -----------------------------------------------------------

void CompressMesh(const std::string& in, const std::string& out)
{
    // ---- Parse glTF ----
    cgltf_options opts = {};
    cgltf_data*   data = nullptr;

    if (cgltf_parse_file(&opts, in.c_str(), &data) != cgltf_result_success) {
        fprintf(stderr, "[MeshCompressor] Failed to parse: %s\n", in.c_str());
        return;
    }
    if (cgltf_load_buffers(&opts, data, in.c_str()) != cgltf_result_success) {
        fprintf(stderr, "[MeshCompressor] Failed to load buffers: %s\n", in.c_str());
        cgltf_free(data);
        return;
    }

    std::string gltfDir = fs::path(in).parent_path().string();
    std::string outDir  = fs::path(out).parent_path().string();
    if (outDir.empty()) outDir = ".";

    // ---- Global geometry arrays ----
    std::vector<MeshVertex>        allVertices;
    std::vector<uint32_t>          allIndices;
    std::vector<MeshMeshlet>       allMeshlets;
    std::vector<uint32_t>          allMeshletVertices;
    std::vector<uint8_t>           allMeshletTriangles;
    std::vector<MeshMeshletBounds> allMeshletBounds;

    std::vector<MeshInstance> instances;
    std::vector<MeshMaterial> materials;

    // ---- Process materials & compress textures ----
    for (size_t i = 0; i < data->materials_count; i++) {
        cgltf_material& mat = data->materials[i];
        MeshMaterial mm = {};

        if (mat.has_pbr_metallic_roughness) {
            cgltf_pbr_metallic_roughness& pbr = mat.pbr_metallic_roughness;
            if (pbr.base_color_texture.texture && pbr.base_color_texture.texture->image)
                ProcessTexture(gltfDir, outDir, pbr.base_color_texture.texture->image->uri,
                               mm.AlbedoPath, sizeof(mm.AlbedoPath));
            if (pbr.metallic_roughness_texture.texture && pbr.metallic_roughness_texture.texture->image)
                ProcessTexture(gltfDir, outDir, pbr.metallic_roughness_texture.texture->image->uri,
                               mm.ORMPath, sizeof(mm.ORMPath));
        }
        if (mat.normal_texture.texture && mat.normal_texture.texture->image)
            ProcessTexture(gltfDir, outDir, mat.normal_texture.texture->image->uri,
                           mm.NormalPath, sizeof(mm.NormalPath));
        if (mat.emissive_texture.texture && mat.emissive_texture.texture->image)
            ProcessTexture(gltfDir, outDir, mat.emissive_texture.texture->image->uri,
                           mm.EmissivePath, sizeof(mm.EmissivePath));

        materials.push_back(mm);
    }

    // ---- Process meshes ----
    constexpr size_t kMaxVerts     = 64;
    constexpr size_t kMaxTriangles = 124;
    constexpr float  kConeWeight   = 0.5f;

    for (size_t mi = 0; mi < data->meshes_count; mi++) {
        cgltf_mesh& mesh = data->meshes[mi];

        for (size_t pi = 0; pi < mesh.primitives_count; pi++) {
            cgltf_primitive& prim = mesh.primitives[pi];
            if (prim.type != cgltf_primitive_type_triangles) continue;

            // ---- Extract vertex attributes (no tangent — we generate them) ----
            cgltf_accessor* posAcc  = nullptr;
            cgltf_accessor* normAcc = nullptr;
            cgltf_accessor* uvAcc   = nullptr;

            for (size_t ai = 0; ai < prim.attributes_count; ai++) {
                cgltf_attribute& attr = prim.attributes[ai];
                switch (attr.type) {
                    case cgltf_attribute_type_position: posAcc  = attr.data; break;
                    case cgltf_attribute_type_normal:   normAcc = attr.data; break;
                    case cgltf_attribute_type_texcoord:
                        if (attr.index == 0) uvAcc = attr.data;
                        break;
                    default: break;
                }
            }

            if (!posAcc) continue;
            size_t rawVertexCount = posAcc->count;

            std::vector<MeshVertex> rawVerts(rawVertexCount);
            for (size_t vi = 0; vi < rawVertexCount; vi++) {
                MeshVertex& v = rawVerts[vi];
                cgltf_accessor_read_float(posAcc,  vi, v.Position, 3);
                if (normAcc) cgltf_accessor_read_float(normAcc, vi, v.Normal, 3);
                if (uvAcc)   cgltf_accessor_read_float(uvAcc,   vi, v.UV,     2);
            }

            // ---- Extract indices ----
            std::vector<uint32_t> rawIndices;
            if (prim.indices) {
                rawIndices.resize(prim.indices->count);
                for (size_t ii = 0; ii < prim.indices->count; ii++)
                    rawIndices[ii] = static_cast<uint32_t>(cgltf_accessor_read_index(prim.indices, ii));
            } else {
                rawIndices.resize(rawVertexCount);
                for (size_t ii = 0; ii < rawVertexCount; ii++)
                    rawIndices[ii] = static_cast<uint32_t>(ii);
            }
            size_t indexCount = rawIndices.size();

            // ---- Expand to unindexed, then generate tangents via MikktSpace ----
            // MikktSpace must not reuse an existing index list — it outputs per-corner
            // tangents. We expand here and let meshopt re-index afterward, which merges
            // vertices that are now bitwise-identical (same pos/normal/uv/tangent).
            std::vector<MeshVertex> expanded(indexCount);
            for (size_t ii = 0; ii < indexCount; ii++)
                expanded[ii] = rawVerts[rawIndices[ii]];

            GenerateTangents(expanded);

            // ---- Re-index with meshopt (deduplication + full optimization) ----
            std::vector<uint32_t> remap(indexCount);
            size_t uniqueCount = meshopt_generateVertexRemap(
                remap.data(), nullptr, indexCount,
                expanded.data(), indexCount, sizeof(MeshVertex));

            std::vector<MeshVertex> optVerts(uniqueCount);
            std::vector<uint32_t>   optIndices(indexCount);
            meshopt_remapVertexBuffer(optVerts.data(), expanded.data(), indexCount, sizeof(MeshVertex), remap.data());
            meshopt_remapIndexBuffer(optIndices.data(), nullptr, indexCount, remap.data());

            // ---- Optimize for GPU ----
            meshopt_optimizeVertexCache(optIndices.data(), optIndices.data(), indexCount, uniqueCount);

            meshopt_optimizeOverdraw(
                optIndices.data(), optIndices.data(), indexCount,
                &optVerts[0].Position[0], uniqueCount, sizeof(MeshVertex), 1.05f);

            {
                std::vector<MeshVertex> fetchVerts(uniqueCount);
                meshopt_optimizeVertexFetch(
                    fetchVerts.data(), optIndices.data(), indexCount,
                    optVerts.data(), uniqueCount, sizeof(MeshVertex));
                optVerts = std::move(fetchVerts);
            }

            // ---- AABB ----
            float aabbMin[3] = {  1e30f,  1e30f,  1e30f };
            float aabbMax[3] = { -1e30f, -1e30f, -1e30f };
            for (const MeshVertex& v : optVerts)
                for (int k = 0; k < 3; k++) {
                    aabbMin[k] = std::min(aabbMin[k], v.Position[k]);
                    aabbMax[k] = std::max(aabbMax[k], v.Position[k]);
                }

            // ---- Build meshlets ----
            size_t maxMeshlets = meshopt_buildMeshletsBound(indexCount, kMaxVerts, kMaxTriangles);

            std::vector<meshopt_Meshlet> msMeshlets(maxMeshlets);
            std::vector<uint32_t>        msVertices(maxMeshlets * kMaxVerts);
            std::vector<uint8_t>         msTriangles(maxMeshlets * kMaxTriangles * 3);

            size_t meshletCount = meshopt_buildMeshlets(
                msMeshlets.data(), msVertices.data(), msTriangles.data(),
                optIndices.data(), indexCount,
                &optVerts[0].Position[0], uniqueCount, sizeof(MeshVertex),
                kMaxVerts, kMaxTriangles, kConeWeight);

            // Trim to actual data
            const meshopt_Meshlet& last = msMeshlets[meshletCount - 1];
            msVertices.resize(last.vertex_offset + last.vertex_count);
            msTriangles.resize(last.triangle_offset + ((last.triangle_count * 3 + 3) & ~3u));

            // Optimize each meshlet for locality
            for (size_t k = 0; k < meshletCount; k++) {
                meshopt_optimizeMeshlet(
                    msVertices.data()  + msMeshlets[k].vertex_offset,
                    msTriangles.data() + msMeshlets[k].triangle_offset,
                    msMeshlets[k].triangle_count,
                    msMeshlets[k].vertex_count);
            }

            // ---- Compute meshlet bounds ----
            std::vector<MeshMeshletBounds> localBounds(meshletCount);
            for (size_t k = 0; k < meshletCount; k++) {
                meshopt_Bounds b = meshopt_computeMeshletBounds(
                    msVertices.data()  + msMeshlets[k].vertex_offset,
                    msTriangles.data() + msMeshlets[k].triangle_offset,
                    msMeshlets[k].triangle_count,
                    &optVerts[0].Position[0], uniqueCount, sizeof(MeshVertex));

                MeshMeshletBounds& mb = localBounds[k];
                memcpy(mb.Center,   b.center,    sizeof(mb.Center));
                mb.Radius = b.radius;
                memcpy(mb.ConeApex, b.cone_apex, sizeof(mb.ConeApex));
                memcpy(mb.ConeAxis, b.cone_axis, sizeof(mb.ConeAxis));
                mb.ConeCutoff = b.cone_cutoff;
                memcpy(mb.ConeAxisS8, b.cone_axis_s8, sizeof(mb.ConeAxisS8));
                mb.ConeCutoffS8 = b.cone_cutoff_s8;
            }

            // ---- Material index ----
            uint32_t matIndex = 0;
            if (prim.material) {
                for (size_t m = 0; m < data->materials_count; m++) {
                    if (&data->materials[m] == prim.material) {
                        matIndex = static_cast<uint32_t>(m);
                        break;
                    }
                }
            }

            // ---- Record instance (offsets into global arrays) ----
            MeshInstance inst = {};
            memcpy(inst.AABBMin, aabbMin, sizeof(aabbMin));
            memcpy(inst.AABBMax, aabbMax, sizeof(aabbMax));
            inst.VertexOffset          = static_cast<uint32_t>(allVertices.size());
            inst.IndexOffset           = static_cast<uint32_t>(allIndices.size());
            inst.MeshletOffset         = static_cast<uint32_t>(allMeshlets.size());
            inst.MeshletVerticesOffset = static_cast<uint32_t>(allMeshletVertices.size());
            inst.MeshletIndicesOffset  = static_cast<uint32_t>(allMeshletTriangles.size());
            inst.MeshletBoundsOffset   = static_cast<uint32_t>(allMeshletBounds.size());
            inst.MaterialIndex         = matIndex;
            instances.push_back(inst);

            // ---- Append to global arrays ----
            allVertices.insert(allVertices.end(), optVerts.begin(), optVerts.end());
            allIndices.insert(allIndices.end(), optIndices.begin(), optIndices.end());

            // Meshlets: remap vertex/triangle offsets to global coordinates
            for (size_t k = 0; k < meshletCount; k++) {
                MeshMeshlet mm = {};
                mm.VertexOffset   = inst.MeshletVerticesOffset + msMeshlets[k].vertex_offset;
                mm.TriangleOffset = inst.MeshletIndicesOffset  + msMeshlets[k].triangle_offset;
                mm.VertexCount    = msMeshlets[k].vertex_count;
                mm.TriangleCount  = msMeshlets[k].triangle_count;
                allMeshlets.push_back(mm);
            }

            // Meshlet vertex indirection: local vertex index -> global vertex index
            for (uint32_t vi : msVertices)
                allMeshletVertices.push_back(inst.VertexOffset + vi);

            allMeshletTriangles.insert(allMeshletTriangles.end(), msTriangles.begin(), msTriangles.end());
            allMeshletBounds.insert(allMeshletBounds.end(), localBounds.begin(), localBounds.end());
        }
    }

    cgltf_free(data);

    // ---- Write binary output ----
    // Layout:
    //   MeshHeader
    //   MeshInstance[InstanceCount]
    //   MeshMaterial[MaterialCount]
    //   uint32 vertexCount   + MeshVertex[vertexCount]
    //   uint32 indexCount    + uint32[indexCount]
    //   uint32 meshletCount  + MeshMeshlet[meshletCount]
    //   uint32 mvCount       + uint32[mvCount]    (meshlet vertex indirection)
    //   uint32 mtBytes       + uint8[mtBytes]     (meshlet triangle indices, padded)
    //   uint32 boundsCount   + MeshMeshletBounds[boundsCount]

    std::ofstream file(out, std::ios::binary);
    if (!file) {
        fprintf(stderr, "[MeshCompressor] Failed to open output: %s\n", out.c_str());
        return;
    }

    auto writeU32  = [&](uint32_t v)              { file.write(reinterpret_cast<const char*>(&v), sizeof(v)); };
    auto writeSpan = [&](const void* p, size_t n) { file.write(reinterpret_cast<const char*>(p), static_cast<std::streamsize>(n)); };

    MeshHeader header = {};
    header.InstanceCount = static_cast<uint32_t>(instances.size());
    header.MaterialCount = static_cast<uint32_t>(materials.size());
    writeSpan(&header, sizeof(header));
    writeSpan(instances.data(), sizeof(MeshInstance) * instances.size());
    writeSpan(materials.data(), sizeof(MeshMaterial) * materials.size());

    writeU32(static_cast<uint32_t>(allVertices.size()));
    writeSpan(allVertices.data(), sizeof(MeshVertex) * allVertices.size());

    writeU32(static_cast<uint32_t>(allIndices.size()));
    writeSpan(allIndices.data(), sizeof(uint32_t) * allIndices.size());

    writeU32(static_cast<uint32_t>(allMeshlets.size()));
    writeSpan(allMeshlets.data(), sizeof(MeshMeshlet) * allMeshlets.size());

    writeU32(static_cast<uint32_t>(allMeshletVertices.size()));
    writeSpan(allMeshletVertices.data(), sizeof(uint32_t) * allMeshletVertices.size());

    writeU32(static_cast<uint32_t>(allMeshletTriangles.size()));
    writeSpan(allMeshletTriangles.data(), allMeshletTriangles.size());

    writeU32(static_cast<uint32_t>(allMeshletBounds.size()));
    writeSpan(allMeshletBounds.data(), sizeof(MeshMeshletBounds) * allMeshletBounds.size());

    printf("[MeshCompressor] %s -> %s | instances: %zu | materials: %zu | verts: %zu | meshlets: %zu\n",
           in.c_str(), out.c_str(),
           instances.size(), materials.size(),
           allVertices.size(), allMeshlets.size());
}
