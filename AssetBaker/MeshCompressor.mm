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
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

// ---- Timing helpers ---------------------------------------------------------

static double NowSeconds()
{
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static std::string FormatDuration(double seconds)
{
    char buf[64];
    if (seconds < 60.0)
        snprintf(buf, sizeof(buf), "%.2fs", seconds);
    else
        snprintf(buf, sizeof(buf), "%dm%02ds", (int)(seconds / 60.0), (int)seconds % 60);
    return buf;
}

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

// ---- Transform helpers ------------------------------------------------------
// cgltf matrices are column-major: M[col*4 + row].

static void TransformPoint(float out[3], const float M[16], const float p[3])
{
    out[0] = M[0]*p[0] + M[4]*p[1] + M[8]*p[2]  + M[12];
    out[1] = M[1]*p[0] + M[5]*p[1] + M[9]*p[2]  + M[13];
    out[2] = M[2]*p[0] + M[6]*p[1] + M[10]*p[2] + M[14];
}

// Transform a direction by the adjugate (det * inverse-transpose) of the
// upper-left 3x3 of M, then renormalize.  Correct for non-uniform scale.
static void TransformNormal(float out[3], const float M[16], const float n[3])
{
    float c0 = M[5]*M[10] - M[9]*M[6];
    float c1 = M[9]*M[2]  - M[1]*M[10];
    float c2 = M[1]*M[6]  - M[5]*M[2];
    float c3 = M[8]*M[6]  - M[4]*M[10];
    float c4 = M[0]*M[10] - M[8]*M[2];
    float c5 = M[4]*M[2]  - M[0]*M[6];
    float c6 = M[4]*M[9]  - M[8]*M[5];
    float c7 = M[8]*M[1]  - M[0]*M[9];
    float c8 = M[0]*M[5]  - M[4]*M[1];
    out[0] = c0*n[0] + c3*n[1] + c6*n[2];
    out[1] = c1*n[0] + c4*n[1] + c7*n[2];
    out[2] = c2*n[0] + c5*n[1] + c8*n[2];
    float len = std::sqrt(out[0]*out[0] + out[1]*out[1] + out[2]*out[2]);
    if (len > 1e-8f) { out[0] /= len; out[1] /= len; out[2] /= len; }
}

// ---- Helpers ----------------------------------------------------------------

static std::string OutputTexturePath(const std::string& outDir, const char* uri)
{
    return outDir + "/" + fs::path(uri).stem().string() + ".tex";
}

// ---- Per-primitive work item ------------------------------------------------
// Filled on the main thread, processed in parallel, merged back in order.

// Per-LOD geometry data for a single primitive.
struct LODGeometry
{
    std::vector<uint32_t>          indices;
    std::vector<MeshMeshlet>       meshlets;
    std::vector<uint32_t>          meshletVertices;
    std::vector<uint8_t>           meshletTriangles;
    std::vector<MeshMeshletBounds> meshletBounds;
};

struct PrimitiveResult
{
    // Vertices are shared across all LODs (simplification reuses original vertices)
    std::vector<MeshVertex>        vertices;

    // Per-LOD index / meshlet data
    LODGeometry lods[kMaxLODs];
    uint32_t    lodCount = 0;   // how many LODs were actually generated (1..kMaxLODs)

    // Instance (offsets will be fixed up when merging into global arrays)
    MeshInstance inst = {};

    // For logging
    std::string nodeName;
    int         nodeIndex    = 0;
    int         primIndex    = 0;
};

// ---- Build meshlets + bounds for a given index buffer -----------------------

static void BuildMeshletsForLOD(
    LODGeometry&                   out,
    const std::vector<uint32_t>&   indices,
    const std::vector<MeshVertex>& vertices)
{
    constexpr size_t kMaxVerts     = 64;
    constexpr size_t kMaxTriangles = 124;
    constexpr float  kConeWeight   = 0.5f;

    size_t indexCount  = indices.size();
    size_t vertexCount = vertices.size();

    size_t maxMeshlets = meshopt_buildMeshletsBound(indexCount, kMaxVerts, kMaxTriangles);

    std::vector<meshopt_Meshlet> msMeshlets(maxMeshlets);
    std::vector<uint32_t>        msVertices(maxMeshlets * kMaxVerts);
    std::vector<uint8_t>         msTriangles(maxMeshlets * kMaxTriangles * 3);

    size_t meshletCount = meshopt_buildMeshlets(
        msMeshlets.data(), msVertices.data(), msTriangles.data(),
        indices.data(), indexCount,
        &vertices[0].Position[0], vertexCount, sizeof(MeshVertex),
        kMaxVerts, kMaxTriangles, kConeWeight);

    if (meshletCount == 0) return;

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

    // Compute meshlet bounds
    std::vector<MeshMeshletBounds> localBounds(meshletCount);
    for (size_t k = 0; k < meshletCount; k++) {
        meshopt_Bounds b = meshopt_computeMeshletBounds(
            msVertices.data()  + msMeshlets[k].vertex_offset,
            msTriangles.data() + msMeshlets[k].triangle_offset,
            msMeshlets[k].triangle_count,
            &vertices[0].Position[0], vertexCount, sizeof(MeshVertex));

        MeshMeshletBounds& mb = localBounds[k];
        memcpy(mb.Center,   b.center,    sizeof(mb.Center));
        mb.Radius = b.radius;
        memcpy(mb.ConeApex, b.cone_apex, sizeof(mb.ConeApex));
        memcpy(mb.ConeAxis, b.cone_axis, sizeof(mb.ConeAxis));
        mb.ConeCutoff = b.cone_cutoff;
        memcpy(mb.ConeAxisS8, b.cone_axis_s8, sizeof(mb.ConeAxisS8));
        mb.ConeCutoffS8 = b.cone_cutoff_s8;
    }

    // Convert raw meshopt meshlets into our MeshMeshlet structs.
    // Offsets here are LOCAL (relative to this LOD's own arrays).
    out.meshlets.reserve(meshletCount);
    for (size_t k = 0; k < meshletCount; k++) {
        MeshMeshlet mm = {};
        mm.VertexOffset   = msMeshlets[k].vertex_offset;
        mm.TriangleOffset = msMeshlets[k].triangle_offset;
        mm.VertexCount    = msMeshlets[k].vertex_count;
        mm.TriangleCount  = msMeshlets[k].triangle_count;
        out.meshlets.push_back(mm);
    }

    out.indices         = indices;   // copy
    out.meshletVertices  = std::move(msVertices);
    out.meshletTriangles = std::move(msTriangles);
    out.meshletBounds    = std::move(localBounds);
}

// ---- Process one primitive --------------------------------------------------

static PrimitiveResult ProcessPrimitive(
    const cgltf_node&      node,
    int                    nodeIndex,
    const cgltf_primitive& prim,
    int                    primIndex,
    uint32_t               matIndex)
{
    PrimitiveResult result;
    result.nodeName  = node.name ? node.name : "(unnamed)";
    result.nodeIndex = nodeIndex;
    result.primIndex = primIndex;

    float worldMat[16];
    cgltf_node_transform_world(&node, worldMat);

    // ---- Extract vertex attributes (no tangent — we generate them) ----
    cgltf_accessor* posAcc  = nullptr;
    cgltf_accessor* normAcc = nullptr;
    cgltf_accessor* uvAcc   = nullptr;

    for (size_t ai = 0; ai < prim.attributes_count; ai++) {
        const cgltf_attribute& attr = prim.attributes[ai];
        switch (attr.type) {
            case cgltf_attribute_type_position: posAcc  = attr.data; break;
            case cgltf_attribute_type_normal:   normAcc = attr.data; break;
            case cgltf_attribute_type_texcoord:
                if (attr.index == 0) uvAcc = attr.data;
                break;
            default: break;
        }
    }

    if (!posAcc)
        return result;

    size_t rawVertexCount = posAcc->count;
    std::vector<MeshVertex> rawVerts(rawVertexCount);
    for (size_t vi = 0; vi < rawVertexCount; vi++) {
        MeshVertex& v = rawVerts[vi];
        cgltf_accessor_read_float(posAcc,  vi, v.Position, 3);
        if (normAcc) cgltf_accessor_read_float(normAcc, vi, v.Normal, 3);
        if (uvAcc)   cgltf_accessor_read_float(uvAcc,   vi, v.UV,     2);
    }

    // ---- Apply node world transform (pre-transform vertices) ----
    for (MeshVertex& v : rawVerts) {
        float tp[3], tn[3];
        TransformPoint(tp, worldMat, v.Position);
        TransformNormal(tn, worldMat, v.Normal);
        v.Position[0] = tp[0]; v.Position[1] = tp[1]; v.Position[2] = tp[2];
        v.Normal[0]   = tn[0]; v.Normal[1]   = tn[1]; v.Normal[2]   = tn[2];
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

    // ---- Build LOD0 meshlets (full detail) ----
    BuildMeshletsForLOD(result.lods[0], optIndices, optVerts);
    result.lodCount = 1;

    // ---- Generate simplified LODs (LOD1..LOD4) ----
    // We simplify the *original* LOD0 indices each time (not chaining LODs)
    // to get the best quality at each level. Stop early if simplification
    // produces no meaningful triangle reduction vs the previous LOD.
    size_t prevIndexCount = indexCount;
    for (uint32_t lod = 1; lod < kMaxLODs; lod++) {
        size_t targetIndexCount = static_cast<size_t>(indexCount * kLODThresholds[lod]);
        // Need at least 3 indices to form a triangle
        if (targetIndexCount < 3) break;

        std::vector<uint32_t> simplified(indexCount);
        float lodError = 0.0f;
        size_t simplifiedCount = meshopt_simplify(
            simplified.data(),
            optIndices.data(), indexCount,
            &optVerts[0].Position[0], uniqueCount, sizeof(MeshVertex),
            targetIndexCount, kLODTargetErrors[lod],
            /* options= */ 0, &lodError);

        // If simplification didn't reduce triangle count meaningfully
        // (less than 10% reduction from previous LOD), stop generating LODs.
        if (simplifiedCount >= prevIndexCount * 95 / 100) break;
        // Need at least one triangle
        if (simplifiedCount < 3) break;

        simplified.resize(simplifiedCount);
        prevIndexCount = simplifiedCount;

        // Build meshlets for this LOD
        BuildMeshletsForLOD(result.lods[lod], simplified, optVerts);

        // Only count this LOD if meshlet generation succeeded
        if (result.lods[lod].meshlets.empty()) break;
        result.lodCount = lod + 1;
    }

    // ---- Pack result ----
    MeshInstance& inst = result.inst;
    memset(&inst, 0, sizeof(inst));
    memcpy(inst.AABBMin, aabbMin, sizeof(aabbMin));
    memcpy(inst.AABBMax, aabbMax, sizeof(aabbMax));
    inst.MaterialIndex = matIndex;

    // Fill per-LOD IndexCount in the instance (offsets are fixed up during merge)
    for (uint32_t lod = 0; lod < result.lodCount; lod++) {
        inst.IndexCount[lod] = static_cast<uint32_t>(result.lods[lod].indices.size());
    }

    result.vertices = std::move(optVerts);

    return result;
}

// ---- Simple thread pool -----------------------------------------------------

class ThreadPool
{
public:
    explicit ThreadPool(size_t threadCount)
    {
        mStop = false;
        for (size_t i = 0; i < threadCount; i++)
            mThreads.emplace_back([this]{ WorkerLoop(); });
    }

    ~ThreadPool()
    {
        {
            std::unique_lock<std::mutex> lock(mMutex);
            mStop = true;
        }
        mCV.notify_all();
        for (auto& t : mThreads)
            t.join();
    }

    // Submit a task and return a future-like token (we use a shared_ptr<bool>
    // here for simplicity; callers wait on mDone below).
    void Submit(std::function<void()> task)
    {
        {
            std::unique_lock<std::mutex> lock(mMutex);
            mQueue.push_back(std::move(task));
        }
        mCV.notify_one();
    }

    // Block until the queue is drained and all workers are idle.
    void WaitAll()
    {
        std::unique_lock<std::mutex> lock(mMutex);
        mDone.wait(lock, [this]{
            return mQueue.empty() && mActiveTasks == 0;
        });
    }

private:
    void WorkerLoop()
    {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mMutex);
                mCV.wait(lock, [this]{ return mStop || !mQueue.empty(); });
                if (mStop && mQueue.empty())
                    return;
                task = std::move(mQueue.front());
                mQueue.pop_front();
                ++mActiveTasks;
            }
            task();
            {
                std::unique_lock<std::mutex> lock(mMutex);
                --mActiveTasks;
            }
            mDone.notify_all();
        }
    }

    std::vector<std::thread>         mThreads;
    std::deque<std::function<void()>> mQueue;
    std::mutex                        mMutex;
    std::condition_variable           mCV;
    std::condition_variable           mDone;
    std::atomic<int>                  mActiveTasks { 0 };
    bool                              mStop;
};

// ---- CompressMesh -----------------------------------------------------------

void CompressMesh(const std::string& in, const std::string& out)
{
    const double totalStart = NowSeconds();

    const unsigned int hwThreads   = std::thread::hardware_concurrency();
    const unsigned int threadCount = std::max(1u, hwThreads);

    printf("[MeshCompressor] ---- Starting compression ----\n");
    printf("[MeshCompressor] Input  : %s\n", in.c_str());
    printf("[MeshCompressor] Output : %s\n", out.c_str());
    printf("[MeshCompressor] Threads: %u (hardware concurrency: %u)\n", threadCount, hwThreads);
    fflush(stdout);

    // ---- Parse glTF ----
    printf("[MeshCompressor] Parsing glTF...\n");
    fflush(stdout);

    double t0 = NowSeconds();

    cgltf_options opts = {};
    cgltf_data*   data = nullptr;

    if (cgltf_parse_file(&opts, in.c_str(), &data) != cgltf_result_success) {
        fprintf(stderr, "[MeshCompressor] ERROR: Failed to parse: %s\n", in.c_str());
        return;
    }
    if (cgltf_load_buffers(&opts, data, in.c_str()) != cgltf_result_success) {
        fprintf(stderr, "[MeshCompressor] ERROR: Failed to load buffers: %s\n", in.c_str());
        cgltf_free(data);
        return;
    }

    printf("[MeshCompressor] glTF parsed in %s — %zu mesh(es), %zu node(s), %zu material(s)\n",
           FormatDuration(NowSeconds() - t0).c_str(),
           data->meshes_count,
           data->nodes_count,
           data->materials_count);
    fflush(stdout);

    std::string gltfDir = fs::path(in).parent_path().string();
    std::string outDir  = fs::path(out).parent_path().string();
    if (outDir.empty()) outDir = ".";

    // ---- Collect unique textures to compress --------------------------------
    // Multiple materials may reference the same URI; we deduplicate so each
    // source image is only compressed once, even across threads.

    printf("[MeshCompressor] Collecting textures...\n");
    fflush(stdout);

    struct TextureJob
    {
        std::string src;
        std::string dst;
    };

    std::vector<TextureJob>  textureJobs;
    std::vector<std::string> seenDst;   // cheap dedup for typically small material counts

    auto maybeAddTexture = [&](const char* uri) {
        if (!uri || uri[0] == '\0') return;
        std::string dst = OutputTexturePath(outDir, uri);
        for (const auto& s : seenDst)
            if (s == dst) return;
        seenDst.push_back(dst);
        textureJobs.push_back({ gltfDir + "/" + uri, dst });
    };

    for (size_t i = 0; i < data->materials_count; i++) {
        const cgltf_material& mat = data->materials[i];
        if (mat.has_pbr_metallic_roughness) {
            const cgltf_pbr_metallic_roughness& pbr = mat.pbr_metallic_roughness;
            if (pbr.base_color_texture.texture && pbr.base_color_texture.texture->image)
                maybeAddTexture(pbr.base_color_texture.texture->image->uri);
            if (pbr.metallic_roughness_texture.texture && pbr.metallic_roughness_texture.texture->image)
                maybeAddTexture(pbr.metallic_roughness_texture.texture->image->uri);
        }
        if (mat.normal_texture.texture && mat.normal_texture.texture->image)
            maybeAddTexture(mat.normal_texture.texture->image->uri);
        if (mat.emissive_texture.texture && mat.emissive_texture.texture->image)
            maybeAddTexture(mat.emissive_texture.texture->image->uri);
    }

    printf("[MeshCompressor] %zu unique texture(s) to compress across %zu material(s)\n",
           textureJobs.size(), data->materials_count);
    fflush(stdout);

    // ---- Compress textures in parallel --------------------------------------

#if !SKIP_TEXTURES
    if (!textureJobs.empty()) {
        printf("[MeshCompressor] Compressing textures on %u thread(s)...\n", threadCount);
        fflush(stdout);

        double texStart = NowSeconds();

        std::atomic<size_t> texDone { 0 };
        std::mutex          logMutex;
        const size_t        texTotal = textureJobs.size();

        ThreadPool texPool(threadCount);

        for (size_t i = 0; i < texTotal; i++) {
            texPool.Submit([&, i]() {
                const TextureJob& job = textureJobs[i];
                double jobStart = NowSeconds();
                CompressTexture(job.src, job.dst);
                double elapsed = NowSeconds() - jobStart;

                size_t done = ++texDone;
                {
                    std::lock_guard<std::mutex> lk(logMutex);
                    printf("[MeshCompressor]   texture [%zu/%zu] (%s) -> %s  (%.2fs)\n",
                           done, texTotal,
                           fs::path(job.src).filename().string().c_str(),
                           fs::path(job.dst).filename().string().c_str(),
                           elapsed);
                    fflush(stdout);
                }
            });
        }

        texPool.WaitAll();

        printf("[MeshCompressor] All textures compressed in %s\n",
               FormatDuration(NowSeconds() - texStart).c_str());
        fflush(stdout);
    }
#endif

    // ---- Build material array -----------------------------------------------
    // Now that all textures exist on disk, fill MeshMaterial with the output paths.

    std::vector<MeshMaterial> materials;
    materials.resize(data->materials_count);

    for (size_t i = 0; i < data->materials_count; i++) {
        const cgltf_material& mat = data->materials[i];
        MeshMaterial& mm = materials[i];

        auto fillPath = [&](char* dst, size_t dstSize, const char* uri) {
            if (!uri || uri[0] == '\0') { dst[0] = '\0'; return; }
            std::string path = OutputTexturePath(outDir, uri);
            snprintf(dst, dstSize, "%s", path.c_str());
        };

        if (mat.has_pbr_metallic_roughness) {
            const cgltf_pbr_metallic_roughness& pbr = mat.pbr_metallic_roughness;
            if (pbr.base_color_texture.texture && pbr.base_color_texture.texture->image)
                fillPath(mm.AlbedoPath, sizeof(mm.AlbedoPath), pbr.base_color_texture.texture->image->uri);
            if (pbr.metallic_roughness_texture.texture && pbr.metallic_roughness_texture.texture->image)
                fillPath(mm.ORMPath, sizeof(mm.ORMPath), pbr.metallic_roughness_texture.texture->image->uri);
        }
        if (mat.normal_texture.texture && mat.normal_texture.texture->image)
            fillPath(mm.NormalPath, sizeof(mm.NormalPath), mat.normal_texture.texture->image->uri);
        if (mat.emissive_texture.texture && mat.emissive_texture.texture->image)
            fillPath(mm.EmissivePath, sizeof(mm.EmissivePath), mat.emissive_texture.texture->image->uri);
    }

    // ---- Collect primitive work items ---------------------------------------

    printf("[MeshCompressor] Collecting primitives...\n");
    fflush(stdout);

    struct PrimitiveJob
    {
        int         nodeIndex;
        int         primIndex;
        uint32_t    matIndex;
    };

    std::vector<PrimitiveJob> primJobs;

    for (size_t ni = 0; ni < data->nodes_count; ni++) {
        const cgltf_node& node = data->nodes[ni];
        if (!node.mesh) continue;

        for (size_t pi = 0; pi < node.mesh->primitives_count; pi++) {
            const cgltf_primitive& prim = node.mesh->primitives[pi];
            if (prim.type != cgltf_primitive_type_triangles) continue;

            uint32_t matIndex = 0;
            if (prim.material) {
                for (size_t m = 0; m < data->materials_count; m++) {
                    if (&data->materials[m] == prim.material) {
                        matIndex = static_cast<uint32_t>(m);
                        break;
                    }
                }
            }

            primJobs.push_back({ (int)ni, (int)pi, matIndex });
        }
    }

    const size_t primTotal = primJobs.size();

    printf("[MeshCompressor] %zu primitive(s) to process across %zu node(s)\n",
           primTotal, data->nodes_count);
    fflush(stdout);

    // ---- Process primitives in parallel -------------------------------------

    printf("[MeshCompressor] Processing primitives on %u thread(s)...\n", threadCount);
    fflush(stdout);

    double meshStart = NowSeconds();

    // Pre-allocate result slots so threads can write at their job index without
    // any synchronisation on the results vector itself.
    std::vector<PrimitiveResult> results(primTotal);

    std::atomic<size_t> primDone { 0 };
    std::mutex          logMutex2;

    ThreadPool meshPool(threadCount);

    for (size_t i = 0; i < primTotal; i++) {
        meshPool.Submit([&, i]() {
            const PrimitiveJob& job = primJobs[i];
            const cgltf_node&   node = data->nodes[job.nodeIndex];

            double jobStart = NowSeconds();
            results[i] = ProcessPrimitive(
                node, job.nodeIndex,
                node.mesh->primitives[job.primIndex], job.primIndex,
                job.matIndex);
            double elapsed = NowSeconds() - jobStart;

            size_t done = ++primDone;
            {
                std::lock_guard<std::mutex> lk(logMutex2);
                printf("[MeshCompressor]   prim [%zu/%zu] node=%d (%s) prim=%d"
                       " | verts=%zu meshlets=%zu lods=%u (%.2fs)\n",
                       done, primTotal,
                       job.nodeIndex,
                       results[i].nodeName.c_str(),
                       job.primIndex,
                       results[i].vertices.size(),
                       results[i].lods[0].meshlets.size(),
                       results[i].lodCount,
                       elapsed);
                fflush(stdout);
            }
        });
    }

    meshPool.WaitAll();

    printf("[MeshCompressor] All primitives processed in %s\n",
           FormatDuration(NowSeconds() - meshStart).c_str());
    fflush(stdout);

    cgltf_free(data);

    // ---- Merge results into global arrays (single-threaded, in order) -------

    printf("[MeshCompressor] Merging results into global geometry arrays...\n");
    fflush(stdout);

    double mergeStart = NowSeconds();

    // Determine global LOD count: maximum across all primitives
    uint32_t globalLODCount = 1;
    for (size_t i = 0; i < primTotal; i++) {
        if (!results[i].vertices.empty())
            globalLODCount = std::max(globalLODCount, results[i].lodCount);
    }

    // Per-LOD global arrays
    std::vector<MeshVertex>        allVertices;
    std::vector<uint32_t>          allIndices[kMaxLODs];
    std::vector<MeshMeshlet>       allMeshlets[kMaxLODs];
    std::vector<uint32_t>          allMeshletVertices[kMaxLODs];
    std::vector<uint8_t>           allMeshletTriangles[kMaxLODs];
    std::vector<MeshMeshletBounds> allMeshletBounds[kMaxLODs];
    std::vector<MeshInstance>      instances;

    for (size_t i = 0; i < primTotal; i++) {
        PrimitiveResult& r = results[i];
        if (r.vertices.empty()) continue;  // primitive was skipped (e.g. no posAcc)

        const uint32_t globalVertexBase = static_cast<uint32_t>(allVertices.size());

        // Fix up instance vertex offset (shared across all LODs)
        r.inst.VertexOffset = globalVertexBase;

        // Vertices are shared across LODs
        allVertices.insert(allVertices.end(), r.vertices.begin(), r.vertices.end());

        // Merge per-LOD data
        for (uint32_t lod = 0; lod < globalLODCount; lod++) {
            // If this primitive doesn't have this LOD level, fall back to the
            // highest LOD it does have (coarsest available).
            uint32_t srcLod = std::min(lod, r.lodCount - 1);
            const LODGeometry& src = r.lods[srcLod];

            const uint32_t globalIndexBase           = static_cast<uint32_t>(allIndices[lod].size());
            const uint32_t globalMeshletBase         = static_cast<uint32_t>(allMeshlets[lod].size());
            const uint32_t globalMeshletVertexBase   = static_cast<uint32_t>(allMeshletVertices[lod].size());
            const uint32_t globalMeshletTriangleBase = static_cast<uint32_t>(allMeshletTriangles[lod].size());
            const uint32_t globalBoundsBase          = static_cast<uint32_t>(allMeshletBounds[lod].size());

            r.inst.IndexOffset[lod]           = globalIndexBase;
            r.inst.IndexCount[lod]            = static_cast<uint32_t>(src.indices.size());
            r.inst.MeshletOffset[lod]         = globalMeshletBase;
            r.inst.MeshletVerticesOffset[lod] = globalMeshletVertexBase;
            r.inst.MeshletIndicesOffset[lod]  = globalMeshletTriangleBase;
            r.inst.MeshletBoundsOffset[lod]   = globalBoundsBase;
            r.inst.MeshletCount[lod]          = static_cast<uint32_t>(src.meshlets.size());

            // Indices
            allIndices[lod].insert(allIndices[lod].end(), src.indices.begin(), src.indices.end());

            // Meshlets: patch local offsets to global coordinates
            for (MeshMeshlet mm : src.meshlets) {
                mm.VertexOffset   += globalMeshletVertexBase;
                mm.TriangleOffset += globalMeshletTriangleBase;
                allMeshlets[lod].push_back(mm);
            }

            // Meshlet vertex indirection: local vertex index -> global vertex index
            for (uint32_t vi : src.meshletVertices)
                allMeshletVertices[lod].push_back(globalVertexBase + vi);

            allMeshletTriangles[lod].insert(allMeshletTriangles[lod].end(),
                                            src.meshletTriangles.begin(), src.meshletTriangles.end());
            allMeshletBounds[lod].insert(allMeshletBounds[lod].end(),
                                         src.meshletBounds.begin(), src.meshletBounds.end());
        }

        instances.push_back(r.inst);
    }

    printf("[MeshCompressor] Merge done in %s — %u LOD level(s)\n",
           FormatDuration(NowSeconds() - mergeStart).c_str(), globalLODCount);
    fflush(stdout);

    // ---- Write binary output ------------------------------------------------
    // Layout (v2 — with LODs):
    //   MeshHeader { InstanceCount, MaterialCount, LODCount }
    //   MeshInstance[InstanceCount]      (per-LOD offsets baked in)
    //   MeshMaterial[MaterialCount]
    //   uint32 vertexCount + MeshVertex[vertexCount]          (shared across LODs)
    //   --- repeated for each LOD level [0..LODCount-1]: ---
    //   uint32 indexCount    + uint32[indexCount]
    //   uint32 meshletCount  + MeshMeshlet[meshletCount]
    //   uint32 mvCount       + uint32[mvCount]
    //   uint32 mtBytes       + uint8[mtBytes]
    //   uint32 boundsCount   + MeshMeshletBounds[boundsCount]

    printf("[MeshCompressor] Writing output file: %s\n", out.c_str());
    fflush(stdout);

    double writeStart = NowSeconds();

    std::ofstream file(out, std::ios::binary);
    if (!file) {
        fprintf(stderr, "[MeshCompressor] ERROR: Failed to open output: %s\n", out.c_str());
        return;
    }

    auto writeU32  = [&](uint32_t v)              { file.write(reinterpret_cast<const char*>(&v), sizeof(v)); };
    auto writeSpan = [&](const void* p, size_t n) { file.write(reinterpret_cast<const char*>(p), static_cast<std::streamsize>(n)); };

    MeshHeader header = {};
    header.InstanceCount = static_cast<uint32_t>(instances.size());
    header.MaterialCount = static_cast<uint32_t>(materials.size());
    header.LODCount      = globalLODCount;
    writeSpan(&header, sizeof(header));
    writeSpan(instances.data(), sizeof(MeshInstance) * instances.size());
    writeSpan(materials.data(), sizeof(MeshMaterial) * materials.size());

    // Vertices (shared across all LODs)
    writeU32(static_cast<uint32_t>(allVertices.size()));
    writeSpan(allVertices.data(), sizeof(MeshVertex) * allVertices.size());

    // Per-LOD geometry arrays
    for (uint32_t lod = 0; lod < globalLODCount; lod++) {
        writeU32(static_cast<uint32_t>(allIndices[lod].size()));
        writeSpan(allIndices[lod].data(), sizeof(uint32_t) * allIndices[lod].size());

        writeU32(static_cast<uint32_t>(allMeshlets[lod].size()));
        writeSpan(allMeshlets[lod].data(), sizeof(MeshMeshlet) * allMeshlets[lod].size());

        writeU32(static_cast<uint32_t>(allMeshletVertices[lod].size()));
        writeSpan(allMeshletVertices[lod].data(), sizeof(uint32_t) * allMeshletVertices[lod].size());

        writeU32(static_cast<uint32_t>(allMeshletTriangles[lod].size()));
        writeSpan(allMeshletTriangles[lod].data(), allMeshletTriangles[lod].size());

        writeU32(static_cast<uint32_t>(allMeshletBounds[lod].size()));
        writeSpan(allMeshletBounds[lod].data(), sizeof(MeshMeshletBounds) * allMeshletBounds[lod].size());
    }

    file.close();

    const double totalElapsed = NowSeconds() - totalStart;

    printf("[MeshCompressor] File written in %s\n",
           FormatDuration(NowSeconds() - writeStart).c_str());
    printf("[MeshCompressor] ---- Compression complete ----\n");
    printf("[MeshCompressor]   Total time   : %s\n",   FormatDuration(totalElapsed).c_str());
    printf("[MeshCompressor]   Instances    : %zu\n",  instances.size());
    printf("[MeshCompressor]   Materials    : %zu\n",  materials.size());
    printf("[MeshCompressor]   LOD levels   : %u\n",   globalLODCount);
    printf("[MeshCompressor]   Vertices     : %zu\n",  allVertices.size());
    for (uint32_t lod = 0; lod < globalLODCount; lod++) {
        printf("[MeshCompressor]   LOD%u indices : %zu, meshlets: %zu\n",
               lod, allIndices[lod].size(), allMeshlets[lod].size());
    }
    printf("[MeshCompressor]   Textures     : %zu\n",  textureJobs.size());
    printf("[MeshCompressor]   %s -> %s\n", in.c_str(), out.c_str());
    fflush(stdout);
}
