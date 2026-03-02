//
//  MeshLoader.swift
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 02/03/2026.
//

import Metal
import simd
import Foundation

// ---- CPU-side instance descriptor ------------------------------------------

struct MeshInstance {
    var aabbMin:               SIMD3<Float>
    var aabbMax:               SIMD3<Float>
    var vertexOffset:          UInt32
    var indexOffset:           UInt32
    var indexCount:            UInt32
    var meshletOffset:         UInt32
    var meshletVerticesOffset: UInt32
    var meshletIndicesOffset:  UInt32
    var meshletBoundsOffset:   UInt32
    var materialIndex:         UInt32
    var meshletCount:          UInt32   // derived: not stored in file, computed post-load
}

// ---- CPU-side material descriptor ------------------------------------------

struct MeshMaterial {
    var albedo:   Texture?
    var normal:   Texture?
    var orm:      Texture?
    var emissive: Texture?
}

// ---- Loaded mesh -----------------------------------------------------------

class Mesh {
    var instances: [MeshInstance] = []
    var materials: [MeshMaterial] = []

    var vertexBuffer:          Buffer!   // MeshVertex[]          48 bytes each
    var indexBuffer:           Buffer!   // uint32[]
    var meshletBuffer:         Buffer!   // MeshMeshlet[]         16 bytes each
    var meshletVerticesBuffer: Buffer!   // uint32[]  (vertex indirection)
    var meshletTrianglesBuffer:Buffer!   // uint8[]   (triangle indices, padded per meshlet)
    var meshletBoundsBuffer:   Buffer!   // MeshMeshletBounds[]   48 bytes each
    var instanceBuffer:        Buffer!   // raw MeshInstance[]    56 bytes each

    var totalMeshletCount: Int = 0
    var totalVertexCount:  Int = 0
    var totalIndexCount:   Int = 0
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
        let slice = data[cursor ..< cursor + count]
        cursor += count
        return Data(slice)
    }
}

// ---- Loader ----------------------------------------------------------------

class MeshLoader {
    // C struct byte strides (must stay in sync with MeshCompressor.h):
    //   MeshVertex:         float3 pos + float3 norm + float2 uv + float4 tan = 48
    //   MeshMeshlet:        4 × uint32                                         = 16
    //   MeshInstance (raw): float3×2 + 8×uint32                                = 56
    //   MeshMaterial (raw): 4 × char[256]                                      = 1024
    //   MeshMeshletBounds:  11 floats + 4 int8                                 = 48
    private static let vertexStride        = 48
    private static let meshletStride       = 16
    private static let instanceStride      = 56
    private static let materialStride      = 1024
    private static let boundsStride        = 48

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

        // ---- Instance bytes — kept verbatim for GPU upload ----
        let rawInstances = r.readSlice(count: instanceCount * instanceStride)

        // ---- Material bytes — parsed for texture paths ----
        let rawMaterials = r.readSlice(count: materialCount * materialStride)

        // ---- Parse CPU-side instances ----
        // C MeshInstance field offsets:
        //   0:  AABBMin[3]            (12 bytes)
        //   12: AABBMax[3]            (12 bytes)
        //   24: VertexOffset          (4 bytes)
        //   28: IndexOffset           (4 bytes)
        //   32: IndexCount            (4 bytes)
        //   36: MeshletOffset         (4 bytes)
        //   40: MeshletVerticesOffset (4 bytes)
        //   44: MeshletIndicesOffset  (4 bytes)
        //   48: MeshletBoundsOffset   (4 bytes)
        //   52: MaterialIndex         (4 bytes)
        var swiftInstances = [MeshInstance]()
        swiftInstances.reserveCapacity(instanceCount)

        rawInstances.withUnsafeBytes { ptr in
            for i in 0 ..< instanceCount {
                let b = i * instanceStride
                func f(_ o: Int) -> Float  { ptr.loadUnaligned(fromByteOffset: b + o, as: Float.self)  }
                func u(_ o: Int) -> UInt32 { ptr.loadUnaligned(fromByteOffset: b + o, as: UInt32.self) }
                swiftInstances.append(MeshInstance(
                    aabbMin:               SIMD3(f(0),  f(4),  f(8)),
                    aabbMax:               SIMD3(f(12), f(16), f(20)),
                    vertexOffset:          u(24),
                    indexOffset:           u(28),
                    indexCount:            u(32),
                    meshletOffset:         u(36),
                    meshletVerticesOffset: u(40),
                    meshletIndicesOffset:  u(44),
                    meshletBoundsOffset:   u(48),
                    materialIndex:         u(52),
                    meshletCount:          0
                ))
            }
        }

        // ---- Parse material paths and load textures ----
        // C MeshMaterial layout within each 1024-byte block:
        //   0:   AlbedoPath   char[256]
        //   256: NormalPath   char[256]
        //   512: ORMPath      char[256]
        //   768: EmissivePath char[256]
        //
        // Texture loading is the bottleneck: progress 5 % → 85 %.
        var swiftMaterials = [MeshMaterial]()
        swiftMaterials.reserveCapacity(materialCount)

        rawMaterials.withUnsafeBytes { ptr in
            for i in 0 ..< materialCount {
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

                swiftMaterials.append(MeshMaterial(
                    albedo:   tex(off: 0,   label: "Albedo[\(i)]"),
                    normal:   tex(off: 256, label: "Normal[\(i)]"),
                    orm:      tex(off: 512, label: "ORM[\(i)]"),
                    emissive: tex(off: 768, label: "Emissive[\(i)]")
                ))
            }
        }

        // ---- Geometry arrays ----
        progress?(0.85, "Parsing geometry…")
        let vertexCount  = Int(r.read(UInt32.self))
        let vertexData   = r.readSlice(count: vertexCount * vertexStride)

        let indexCount   = Int(r.read(UInt32.self))
        let indexData    = r.readSlice(count: indexCount * MemoryLayout<UInt32>.size)

        let meshletCount = Int(r.read(UInt32.self))
        let meshletData  = r.readSlice(count: meshletCount * meshletStride)

        let mvCount      = Int(r.read(UInt32.self))
        let mvData       = r.readSlice(count: mvCount * MemoryLayout<UInt32>.size)

        let mtBytes      = Int(r.read(UInt32.self))
        let mtData       = r.readSlice(count: mtBytes)

        let boundsCount  = Int(r.read(UInt32.self))
        let boundsData   = r.readSlice(count: boundsCount * boundsStride)

        // ---- Derive per-instance meshlet count ----
        for i in 0 ..< instanceCount {
            let next = (i + 1 < instanceCount)
                ? swiftInstances[i + 1].meshletOffset
                : UInt32(meshletCount)
            swiftInstances[i].meshletCount = next - swiftInstances[i].meshletOffset
        }

        // ---- Build GPU buffers ----
        progress?(0.90, "Uploading to GPU…")
        let mesh                   = Mesh()
        mesh.instances             = swiftInstances
        mesh.materials             = swiftMaterials
        mesh.totalMeshletCount     = meshletCount
        mesh.totalVertexCount      = vertexCount
        mesh.totalIndexCount       = indexCount

        let name = url.deletingPathExtension().lastPathComponent

        vertexData.withUnsafeBytes   { mesh.vertexBuffer           = Buffer(bytes: $0.baseAddress!, size: vertexData.count,   makeResidentNow: false) }
        indexData.withUnsafeBytes    { mesh.indexBuffer            = Buffer(bytes: $0.baseAddress!, size: indexData.count,    makeResidentNow: false) }
        meshletData.withUnsafeBytes  { mesh.meshletBuffer          = Buffer(bytes: $0.baseAddress!, size: meshletData.count,  makeResidentNow: false) }
        mvData.withUnsafeBytes       { mesh.meshletVerticesBuffer  = Buffer(bytes: $0.baseAddress!, size: mvData.count,       makeResidentNow: false) }
        mtData.withUnsafeBytes       { mesh.meshletTrianglesBuffer = Buffer(bytes: $0.baseAddress!, size: mtData.count,       makeResidentNow: false) }
        boundsData.withUnsafeBytes   { mesh.meshletBoundsBuffer    = Buffer(bytes: $0.baseAddress!, size: boundsData.count,   makeResidentNow: false) }
        rawInstances.withUnsafeBytes { mesh.instanceBuffer         = Buffer(bytes: $0.baseAddress!, size: rawInstances.count, makeResidentNow: false) }

        mesh.vertexBuffer.setName(name:           "\(name) Vertices")
        mesh.indexBuffer.setName(name:            "\(name) Indices")
        mesh.meshletBuffer.setName(name:          "\(name) Meshlets")
        mesh.meshletVerticesBuffer.setName(name:  "\(name) Meshlet Vertices")
        mesh.meshletTrianglesBuffer.setName(name: "\(name) Meshlet Triangles")
        mesh.meshletBoundsBuffer.setName(name:    "\(name) Meshlet Bounds")
        mesh.instanceBuffer.setName(name:         "\(name) Instances")

        // ---- Commit residency atomically on the calling thread ----
        // All seven buffers are added to the residency set in one go, avoiding
        // any race if load() is called from a background thread while the main
        // thread is already committing a prior residency set update.
        mesh.vertexBuffer.makeResident()
        mesh.indexBuffer.makeResident()
        mesh.meshletBuffer.makeResident()
        mesh.meshletVerticesBuffer.makeResident()
        mesh.meshletTrianglesBuffer.makeResident()
        mesh.meshletBoundsBuffer.makeResident()
        mesh.instanceBuffer.makeResident()

        progress?(1.00, "Done")
        print("[MeshLoader] \(name): \(instanceCount) instance(s), \(materialCount) material(s), \(vertexCount) verts, \(meshletCount) meshlets")
        return mesh
    }
}
