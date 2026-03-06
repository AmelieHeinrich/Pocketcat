//
//  MeshLoader.swift
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 02/03/2026.
//

import Foundation
import Metal
import simd

// ---- Constants matching MeshCompressor.h -----------------------------------

let kMaxLODs: Int = 5

// ---- CPU-side instance descriptor ------------------------------------------

struct MeshInstance {
    var aabbMin: SIMD3<Float>
    var aabbMax: SIMD3<Float>
    var vertexOffset: UInt32
    var materialIndex: UInt32

    // Per-LOD offsets and counts (up to kMaxLODs entries; only lodCount are valid)
    var indexOffset: [UInt32]  // [kMaxLODs]
    var indexCount: [UInt32]  // [kMaxLODs]
    var meshletOffset: [UInt32]  // [kMaxLODs]
    var meshletVerticesOffset: [UInt32]  // [kMaxLODs]
    var meshletIndicesOffset: [UInt32]  // [kMaxLODs]
    var meshletBoundsOffset: [UInt32]  // [kMaxLODs]
    var meshletCount: [UInt32]  // [kMaxLODs]
}

// ---- CPU-side material descriptor ------------------------------------------

struct MeshMaterial {
    var albedo: Texture?
    var normal: Texture?
    var orm: Texture?
    var emissive: Texture?
}

// ---- Per-LOD GPU buffers ---------------------------------------------------

class MeshLOD {
    var indexBuffer: Buffer!
    var meshletBuffer: Buffer!
    var meshletVerticesBuffer: Buffer!
    var meshletTrianglesBuffer: Buffer!
    var meshletBoundsBuffer: Buffer!

    var totalMeshletCount: Int = 0
    var totalIndexCount: Int = 0
}

// ---- Loaded mesh -----------------------------------------------------------

class Mesh {
    var instances: [MeshInstance] = []
    var materials: [MeshMaterial] = []

    var lodCount: Int = 1

    // Vertex buffer is shared across all LODs
    var vertexBuffer: Buffer!
    var totalVertexCount: Int = 0

    // Per-LOD geometry buffers
    var lods: [MeshLOD] = []

    // Instance buffer (raw bytes uploaded to GPU)
    var instanceBuffer: Buffer!

    var blas: BLAS!

    // Legacy accessors for LOD0 (convenience for code that doesn't care about LODs)
    var indexBuffer: Buffer! { lods.isEmpty ? nil : lods[0].indexBuffer }
    var meshletBuffer: Buffer! { lods.isEmpty ? nil : lods[0].meshletBuffer }
    var meshletVerticesBuffer: Buffer! { lods.isEmpty ? nil : lods[0].meshletVerticesBuffer }
    var meshletTrianglesBuffer: Buffer! { lods.isEmpty ? nil : lods[0].meshletTrianglesBuffer }
    var meshletBoundsBuffer: Buffer! { lods.isEmpty ? nil : lods[0].meshletBoundsBuffer }
    var totalMeshletCount: Int { lods.isEmpty ? 0 : lods[0].totalMeshletCount }
    var totalIndexCount: Int { lods.isEmpty ? 0 : lods[0].totalIndexCount }
}

// ---- Sequential binary reader ----------------------------------------------

private final class BinaryReader {
    private let data: Data
    private var cursor: Int = 0

    init(_ data: Data) { self.data = data }

    func read<T>(_ type: T.Type = T.self) -> T {
        let value = data.withUnsafeBytes {
            $0.loadUnaligned(fromByteOffset: cursor, as: T.self)
        }
        cursor += MemoryLayout<T>.size
        return value
    }

    func readSlice(count: Int) -> Data {
        let slice = data[cursor..<cursor + count]
        cursor += count
        return Data(slice)
    }
}

// ---- Loader ----------------------------------------------------------------

class MeshLoader {
    // C struct byte strides (must stay in sync with MeshCompressor.h):
    //   MeshVertex:         float3 pos + float3 norm + float2 uv + float4 tan = 48
    //   MeshMeshlet:        4 × uint32                                         = 16
    //   MeshMeshletBounds:  11 floats + 4 int8                                 = 48
    //   MeshMaterial (raw): 4 × char[256]                                      = 1024
    //
    // MeshInstance (raw) — v2 with per-LOD arrays:
    //   float3 AABBMin                   12
    //   float3 AABBMax                   12
    //   uint32 VertexOffset               4
    //   uint32 MaterialIndex              4
    //   uint32 IndexOffset[kMaxLODs]     20
    //   uint32 IndexCount[kMaxLODs]      20
    //   uint32 MeshletOffset[kMaxLODs]   20
    //   uint32 MeshletVerticesOffset[kMaxLODs] 20
    //   uint32 MeshletIndicesOffset[kMaxLODs]  20
    //   uint32 MeshletBoundsOffset[kMaxLODs]   20
    //   uint32 MeshletCount[kMaxLODs]    20
    //                                   ----
    //                                    172
    //
    // MeshHeader:
    //   uint32 InstanceCount              4
    //   uint32 MaterialCount              4
    //   uint32 LODCount                   4
    //                                   ----
    //                                     12
    private static let vertexStride = 48
    private static let meshletStride = 16
    private static let instanceStride = 172
    private static let materialStride = 1024
    private static let boundsStride = 48

    /// - Parameters:
    ///   - url: Location of the `.bin` scene file.
    ///   - progress: Optional callback invoked from the calling thread with
    ///     a value in [0, 1] and a human-readable status string.
    static func load(url: URL, progress: ((Double, String) -> Void)? = nil) -> Mesh? {
        progress?(0.00, "Reading scene file…")
        guard let data = try? Data(contentsOf: url, options: .mappedIfSafe) else {
            print("[MeshLoader] Failed to read: \(url.path)")
            return nil
        }

        let r = BinaryReader(data)

        // ---- Header ----
        progress?(0.02, "Parsing scene header…")
        let instanceCount = Int(r.read(UInt32.self))
        let materialCount = Int(r.read(UInt32.self))
        let lodCount = Int(r.read(UInt32.self))

        let clampedLODCount = min(max(lodCount, 1), kMaxLODs)

        // ---- Instance bytes — kept verbatim for GPU upload ----
        let rawInstances = r.readSlice(count: instanceCount * instanceStride)

        // ---- Material bytes — parsed for texture paths ----
        let rawMaterials = r.readSlice(count: materialCount * materialStride)

        // ---- Parse CPU-side instances ----
        // C MeshInstance field offsets (v2):
        //   0:  AABBMin[3]            (12 bytes)
        //   12: AABBMax[3]            (12 bytes)
        //   24: VertexOffset          (4 bytes)
        //   28: MaterialIndex         (4 bytes)
        //   32: IndexOffset[5]        (20 bytes)
        //   52: IndexCount[5]         (20 bytes)
        //   72: MeshletOffset[5]      (20 bytes)
        //   92: MeshletVerticesOffset[5]  (20 bytes)
        //  112: MeshletIndicesOffset[5]   (20 bytes)
        //  132: MeshletBoundsOffset[5]    (20 bytes)
        //  152: MeshletCount[5]       (20 bytes)
        var swiftInstances = [MeshInstance]()
        swiftInstances.reserveCapacity(instanceCount)

        rawInstances.withUnsafeBytes { ptr in
            for i in 0..<instanceCount {
                let b = i * instanceStride
                func f(_ o: Int) -> Float {
                    ptr.loadUnaligned(fromByteOffset: b + o, as: Float.self)
                }
                func u(_ o: Int) -> UInt32 {
                    ptr.loadUnaligned(fromByteOffset: b + o, as: UInt32.self)
                }

                func uArray(_ baseOff: Int) -> [UInt32] {
                    (0..<kMaxLODs).map { u(baseOff + $0 * 4) }
                }

                swiftInstances.append(
                    MeshInstance(
                        aabbMin: SIMD3(f(0), f(4), f(8)),
                        aabbMax: SIMD3(f(12), f(16), f(20)),
                        vertexOffset: u(24),
                        materialIndex: u(28),
                        indexOffset: uArray(32),
                        indexCount: uArray(52),
                        meshletOffset: uArray(72),
                        meshletVerticesOffset: uArray(92),
                        meshletIndicesOffset: uArray(112),
                        meshletBoundsOffset: uArray(132),
                        meshletCount: uArray(152)
                    ))
            }
        }

        // ---- Parse material paths and load textures ----
        var swiftMaterials = [MeshMaterial]()
        swiftMaterials.reserveCapacity(materialCount)

        rawMaterials.withUnsafeBytes { ptr in
            for i in 0..<materialCount {
                let base = i * materialStride

                let pct = 0.05 + 0.80 * Double(i) / Double(max(materialCount, 1))
                progress?(pct, "Loading material \(i + 1) of \(materialCount)…")

                func path(off: Int) -> String? {
                    let raw = Data(bytes: ptr.baseAddress!.advanced(by: base + off), count: 256)
                    let end = raw.firstIndex(of: 0) ?? raw.endIndex
                    let s = String(bytes: raw[..<end], encoding: .utf8) ?? ""
                    return s.isEmpty ? nil : s
                }

                func tex(off: Int, label: String) -> Texture? {
                    guard let p = path(off: off) else { return nil }
                    let fileURL = URL(fileURLWithPath: p)
                    return TextureLoader.load(
                        resource: fileURL.deletingPathExtension().lastPathComponent,
                        withExtension: fileURL.pathExtension,
                        label: label
                    )
                }

                swiftMaterials.append(
                    MeshMaterial(
                        albedo: tex(off: 0, label: "Albedo[\(i)]"),
                        normal: tex(off: 256, label: "Normal[\(i)]"),
                        orm: tex(off: 512, label: "ORM[\(i)]"),
                        emissive: tex(off: 768, label: "Emissive[\(i)]")
                    ))
            }
        }

        // ---- Shared vertex array ----
        progress?(0.85, "Parsing geometry…")
        let vertexCount = Int(r.read(UInt32.self))
        let vertexData = r.readSlice(count: vertexCount * vertexStride)

        // ---- Per-LOD geometry arrays ----
        struct LODData {
            var indexCount: Int
            var indexData: Data
            var meshletCount: Int
            var meshletData: Data
            var mvCount: Int
            var mvData: Data
            var mtBytes: Int
            var mtData: Data
            var boundsCount: Int
            var boundsData: Data
        }

        var lodDataArray = [LODData]()
        lodDataArray.reserveCapacity(clampedLODCount)

        for lod in 0..<clampedLODCount {
            progress?(
                0.85 + 0.05 * Double(lod) / Double(clampedLODCount), "Parsing LOD\(lod) geometry…")

            let idxCount = Int(r.read(UInt32.self))
            let idxData = r.readSlice(count: idxCount * MemoryLayout<UInt32>.size)

            let mlCount = Int(r.read(UInt32.self))
            let mlData = r.readSlice(count: mlCount * meshletStride)

            let mvC = Int(r.read(UInt32.self))
            let mvD = r.readSlice(count: mvC * MemoryLayout<UInt32>.size)

            let mtB = Int(r.read(UInt32.self))
            let mtD = r.readSlice(count: mtB)

            let bnC = Int(r.read(UInt32.self))
            let bnD = r.readSlice(count: bnC * boundsStride)

            lodDataArray.append(
                LODData(
                    indexCount: idxCount, indexData: idxData,
                    meshletCount: mlCount, meshletData: mlData,
                    mvCount: mvC, mvData: mvD,
                    mtBytes: mtB, mtData: mtD,
                    boundsCount: bnC, boundsData: bnD
                ))
        }

        // ---- Build GPU buffers ----
        progress?(0.90, "Uploading to GPU…")
        let mesh = Mesh()
        mesh.instances = swiftInstances
        mesh.materials = swiftMaterials
        mesh.lodCount = clampedLODCount
        mesh.totalVertexCount = vertexCount

        let name = url.deletingPathExtension().lastPathComponent

        // Shared vertex buffer
        vertexData.withUnsafeBytes {
            mesh.vertexBuffer = Buffer(
                bytes: $0.baseAddress!, size: vertexData.count, makeResidentNow: false)
        }
        mesh.vertexBuffer.setName(name: "\(name) Vertices")

        // Instance buffer
        rawInstances.withUnsafeBytes {
            mesh.instanceBuffer = Buffer(
                bytes: $0.baseAddress!, size: rawInstances.count, makeResidentNow: false)
        }
        mesh.instanceBuffer.setName(name: "\(name) Instances")

        // Per-LOD buffers
        for lod in 0..<clampedLODCount {
            let ld = lodDataArray[lod]
            let meshLOD = MeshLOD()
            meshLOD.totalMeshletCount = ld.meshletCount
            meshLOD.totalIndexCount = ld.indexCount

            ld.indexData.withUnsafeBytes {
                meshLOD.indexBuffer = Buffer(
                    bytes: $0.baseAddress!, size: ld.indexData.count, makeResidentNow: false)
            }
            ld.meshletData.withUnsafeBytes {
                meshLOD.meshletBuffer = Buffer(
                    bytes: $0.baseAddress!, size: ld.meshletData.count, makeResidentNow: false)
            }
            ld.mvData.withUnsafeBytes {
                meshLOD.meshletVerticesBuffer = Buffer(
                    bytes: $0.baseAddress!, size: ld.mvData.count, makeResidentNow: false)
            }
            ld.mtData.withUnsafeBytes {
                meshLOD.meshletTrianglesBuffer = Buffer(
                    bytes: $0.baseAddress!, size: ld.mtData.count, makeResidentNow: false)
            }
            ld.boundsData.withUnsafeBytes {
                meshLOD.meshletBoundsBuffer = Buffer(
                    bytes: $0.baseAddress!, size: ld.boundsData.count, makeResidentNow: false)
            }

            meshLOD.indexBuffer.setName(name: "\(name) LOD\(lod) Indices")
            meshLOD.meshletBuffer.setName(name: "\(name) LOD\(lod) Meshlets")
            meshLOD.meshletVerticesBuffer.setName(name: "\(name) LOD\(lod) Meshlet Vertices")
            meshLOD.meshletTrianglesBuffer.setName(name: "\(name) LOD\(lod) Meshlet Triangles")
            meshLOD.meshletBoundsBuffer.setName(name: "\(name) LOD\(lod) Meshlet Bounds")

            mesh.lods.append(meshLOD)
        }

        // ---- Commit residency ----
        // if RendererData.device.supportsFamily(.apple9) {
        //     mesh.blas = BLAS(model: mesh, makeResidentNow: false)
        //     mesh.blas.setName(name: "\(name) BLAS")
        // }

        mesh.vertexBuffer.makeResident()
        mesh.instanceBuffer.makeResident()
        for lod in mesh.lods {
            lod.indexBuffer.makeResident()
            lod.meshletBuffer.makeResident()
            lod.meshletVerticesBuffer.makeResident()
            lod.meshletTrianglesBuffer.makeResident()
            lod.meshletBoundsBuffer.makeResident()
        }
        // mesh.blas?.makeResident()

        // progress?(0.95, "Building BLAS…")
        // if RendererData.device.supportsFamily(.apple9) {
        //     let commandBuffer = CommandBuffer()
        //     commandBuffer.begin(commitResidencySet: false)
        //     let cp = commandBuffer.beginComputePass()
        //     cp.buildBLAS(blas: mesh.blas)
        //     cp.end()
        //     commandBuffer.end()
        //     commandBuffer.commit()
//
        //     RendererData.waitIdle()
//
        //     mesh.blas.destroyScratch()
        // }

        progress?(1.00, "Done")
        print(
            "[MeshLoader] \(name): \(instanceCount) instance(s), \(materialCount) material(s), \(vertexCount) verts, \(clampedLODCount) LOD(s)"
        )
        for lod in 0..<clampedLODCount {
            print(
                "[MeshLoader]   LOD\(lod): \(lodDataArray[lod].indexCount) indices, \(lodDataArray[lod].meshletCount) meshlets"
            )
        }
        return mesh
    }
}
