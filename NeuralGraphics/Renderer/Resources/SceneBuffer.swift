//
//  SceneBuffer.swift
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 2026.
//

import Metal
import simd

private struct GPUSceneCamera {
    var view: simd_float4x4
    var projection: simd_float4x4
    var viewProjection: simd_float4x4
    var inverseView: simd_float4x4
    var inverseProjection: simd_float4x4
    var inverseViewProjection: simd_float4x4
    var position: simd_float4
    var direction: simd_float4
}

private struct GPUSceneMaterial {
    var albedoID: UInt64
    var normalID: UInt64
    var ormID: UInt64
    var emissiveID: UInt64
    var flags: UInt32
    var alphaMode: UInt32
}

private struct GPUSceneInstanceLOD {
    var indexBuffer: UInt64
    var meshlets: UInt64
    var meshletVertices: UInt64
    var meshletTriangles: UInt64
    var meshletBounds: UInt64
    var indexCount: UInt32
    var meshletCount: UInt32
}

private struct GPUSceneInstance {
    var vertexBuffer: UInt64
    var materialIndex: UInt32
    var entityIndex: UInt32
    var lodCount: UInt32
    var aabbMin: simd_float3
    var aabbMax: simd_float3
    var lod0: GPUSceneInstanceLOD
    var lod1: GPUSceneInstanceLOD
    var lod2: GPUSceneInstanceLOD
    var lod3: GPUSceneInstanceLOD
    var lod4: GPUSceneInstanceLOD
}

private struct GPUSceneEntity {
    var transform: simd_float4x4
}

private struct GPUSceneBufferHeader {
    var materialsPtr: UInt64
    var instancesPtr: UInt64
    var entitiesPtr: UInt64
    var asPtr: UInt64

    var camera: GPUSceneCamera
    var materialCount: UInt32
    var instanceCount: UInt32
    var entityCount: UInt32
}

class SceneBufferBuilder {

    // The main GPU buffer holding the entire scene description
    private(set) var buffer: Buffer!
    private(set) var accelerationStructure: TLAS!

    // Sub-buffers for the arrays (so we can grow them independently)
    private(set) var materialsBuffer: Buffer!
    private(set) var instancesBuffer: Buffer!
    private(set) var entitiesBuffer: Buffer!

    // Fallback 1x1 white texture for materials without textures
    private let fallbackTexture: Texture

    // Cached counts
    private(set) var materialCount: Int = 0
    private(set) var instanceCount: Int = 0
    private(set) var entityCount: Int = 0

    init() {
        // Create 1x1 white fallback texture
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false)
        desc.storageMode = .shared
        desc.usage = .shaderRead
        let tex = Texture(descriptor: desc)
        tex.setLabel(name: "Scene Fallback White")
        var px: UInt32 = 0xFFFF_FFFF
        withUnsafePointer(to: &px) {
            tex.uploadData(
                region: MTLRegionMake2D(0, 0, 1, 1), mip: 0,
                data: UnsafeRawPointer($0), bpp: 4)
        }
        self.fallbackTexture = tex
    }

    /// Rebuilds the entire scene buffer from the current RenderScene.
    /// Call this when the scene changes (entities added/removed, meshes loaded).
    func build(scene: RenderScene) {
        // ---- Count everything ----
        entityCount = scene.entities.count
        accelerationStructure = scene.tlas

        var totalMaterials = 0
        var totalInstances = 0
        // Track material offset per entity so we can remap material indices
        var entityMaterialOffsets: [Int] = []
        for entity in scene.entities {
            entityMaterialOffsets.append(totalMaterials)
            totalMaterials += entity.mesh.materials.count
            totalInstances += entity.mesh.instances.count
        }
        materialCount = totalMaterials
        instanceCount = totalInstances

        // ---- Build materials array ----
        let matStride = MemoryLayout<GPUSceneMaterial>.stride
        materialsBuffer = Buffer(size: max(matStride * materialCount, matStride))
        materialsBuffer.setName(name: "Scene Materials")

        let matPtr = materialsBuffer.contents().bindMemory(
            to: GPUSceneMaterial.self,
            capacity: max(materialCount, 1))
        var matIdx = 0
        for entity in scene.entities {
            for mat in entity.mesh.materials {
                var flags: UInt32 = 0
                let albedoID = textureID(mat.albedo, flag: 1, flags: &flags)
                let normalID = textureID(mat.normal, flag: 2, flags: &flags)
                let ormID = textureID(mat.orm, flag: 4, flags: &flags)
                let emissiveID = textureID(mat.emissive, flag: 8, flags: &flags)

                // AlphaMode: 0 = opaque, 1 = mask, 2 = blend
                let alphaMode = mat.alphaMode
                if alphaMode == 0 {
                    flags |= 16  // MaterialFlag_IsOpaque
                }

                matPtr[matIdx] = GPUSceneMaterial(
                    albedoID: albedoID, normalID: normalID,
                    ormID: ormID, emissiveID: emissiveID,
                    flags: flags, alphaMode: alphaMode)
                matIdx += 1
            }
        }

        // ---- Build entities array ----
        let entStride = MemoryLayout<GPUSceneEntity>.stride
        entitiesBuffer = Buffer(size: max(entStride * entityCount, entStride))
        entitiesBuffer.setName(name: "Scene Entities")

        let entPtr = entitiesBuffer.contents().bindMemory(
            to: GPUSceneEntity.self,
            capacity: max(entityCount, 1))
        for (i, entity) in scene.entities.enumerated() {
            entPtr[i] = GPUSceneEntity(transform: entity.transform)
        }

        // ---- Build instances array ----
        let instStride = MemoryLayout<GPUSceneInstance>.stride
        instancesBuffer = Buffer(size: max(instStride * instanceCount, instStride))
        instancesBuffer.setName(name: "Scene Instances")

        let instPtr = instancesBuffer.contents().bindMemory(
            to: GPUSceneInstance.self,
            capacity: max(instanceCount, 1))
        var instIdx = 0
        for (entityIdx, entity) in scene.entities.enumerated() {
            let mesh = entity.mesh
            let vertexBase = mesh.vertexBuffer.getAddress()
            let materialOffset = UInt32(entityMaterialOffsets[entityIdx])

            for instance in mesh.instances {
                let vertexAddr = vertexBase + UInt64(instance.vertexOffset) * 48  // MeshVertex stride

                var lods: [GPUSceneInstanceLOD] = []
                for lod in 0..<kMaxLODs {
                    if lod < mesh.lodCount {
                        let lodData = mesh.lods[lod]

                        // Meshlet stride = 16 bytes (MeshMeshlet)
                        // MeshMeshletBounds stride = 48 bytes
                        let indexAddr =
                            lodData.indexBuffer.getAddress()
                            + UInt64(instance.indexOffset[lod]) * 4
                        let meshletAddr =
                            lodData.meshletBuffer.getAddress()
                            + UInt64(instance.meshletOffset[lod]) * 16
                        let mvAddr = lodData.meshletVerticesBuffer.getAddress()
                        let mtAddr = lodData.meshletTrianglesBuffer.getAddress()
                        let boundsAddr =
                            lodData.meshletBoundsBuffer.getAddress()
                            + UInt64(instance.meshletBoundsOffset[lod]) * 48

                        lods.append(
                            GPUSceneInstanceLOD(
                                indexBuffer: indexAddr,
                                meshlets: meshletAddr,
                                meshletVertices: mvAddr,
                                meshletTriangles: mtAddr,
                                meshletBounds: boundsAddr,
                                indexCount: instance.indexCount[lod],
                                meshletCount: instance.meshletCount[lod]))
                    } else {
                        lods.append(
                            GPUSceneInstanceLOD(
                                indexBuffer: 0, meshlets: 0,
                                meshletVertices: 0, meshletTriangles: 0, meshletBounds: 0,
                                indexCount: 0, meshletCount: 0))
                    }
                }

                instPtr[instIdx] = GPUSceneInstance(
                    vertexBuffer: vertexAddr,
                    materialIndex: instance.materialIndex + materialOffset,
                    entityIndex: UInt32(entityIdx),
                    lodCount: UInt32(mesh.lodCount),
                    aabbMin: instance.aabbMin,
                    aabbMax: instance.aabbMax,
                    lod0: lods[0], lod1: lods[1], lod2: lods[2],
                    lod3: lods[3], lod4: lods[4])
                instIdx += 1
            }
        }

        // ---- Build root header ----
        let headerSize = MemoryLayout<GPUSceneBufferHeader>.stride
        buffer = Buffer(size: headerSize)
        buffer.setName(name: "Scene Buffer")

        updateCamera(CameraData())  // Initialize with identity camera
        updatePointers()
    }

    /// Updates just the camera data in the scene buffer. Call every frame.
    func updateCamera(_ cam: CameraData) {
        guard let buf = buffer else { return }
        let ptr = buf.contents().bindMemory(to: GPUSceneBufferHeader.self, capacity: 1)
        ptr.pointee.camera = GPUSceneCamera(
            view: cam.view,
            projection: cam.projection,
            viewProjection: cam.viewProjection,
            inverseView: cam.inverseView,
            inverseProjection: cam.inverseProjection,
            inverseViewProjection: cam.inverseViewProjection,
            position: SIMD4<Float>(cam.position, cam.near),
            direction: SIMD4<Float>(cam.direction, cam.far))
    }

    /// Updates a single entity's transform. Use for per-frame animation.
    func updateEntityTransform(_ entityIndex: Int, transform: simd_float4x4) {
        guard let buf = entitiesBuffer, entityIndex < entityCount else { return }
        let ptr = buf.contents().bindMemory(to: GPUSceneEntity.self, capacity: entityCount)
        ptr[entityIndex].transform = transform
    }

    /// Returns the GPU address of the root SceneBuffer.
    /// Bind this as a single pointer in any shader stage.
    func getAddress() -> UInt64 {
        return buffer.getAddress()
    }

    // MARK: - Private Helpers

    private func updatePointers() {
        guard let buf = buffer else { return }
        let ptr = buf.contents().bindMemory(to: GPUSceneBufferHeader.self, capacity: 1)
        ptr.pointee.materialCount = UInt32(materialCount)
        ptr.pointee.instanceCount = UInt32(instanceCount)
        ptr.pointee.entityCount = UInt32(entityCount)
        ptr.pointee.asPtr = accelerationStructure.getResourceID()
        ptr.pointee.materialsPtr = materialsBuffer.getAddress()
        ptr.pointee.instancesPtr = instancesBuffer.getAddress()
        ptr.pointee.entitiesPtr = entitiesBuffer.getAddress()
    }

    private func textureID(_ tex: Texture?, flag: UInt32, flags: inout UInt32) -> UInt64 {
        if let t = tex {
            flags |= flag
            return t.texture.gpuResourceID._impl
        }
        return fallbackTexture.texture.gpuResourceID._impl
    }
}
