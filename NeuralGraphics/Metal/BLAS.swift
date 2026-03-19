//
//  BLAS.swift
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 02/03/2026.
//

import Metal

struct PackedVertex {
    var px, py, pz, nx, ny, nz, u, v, tx, ty, tz, tw: Float
}

class BLAS {
    var geometries: [MTL4AccelerationStructureTriangleGeometryDescriptor] = []
    var descriptor: MTL4PrimitiveAccelerationStructureDescriptor
    var accelerationStructure: MTLAccelerationStructure
    var scratchBuffer: Buffer!
    private var allocated: Bool = false

    init(mesh: Mesh, instance: MeshInstance, makeResidentNow: Bool = true) {
        let geometry = MTL4AccelerationStructureTriangleGeometryDescriptor()
        geometry.vertexBuffer = MTL4BufferRangeMake(
            mesh.vertexBuffer.getAddress() + UInt64(instance.vertexOffset)
                * UInt64(MemoryLayout<PackedVertex>.size),
            UInt64(mesh.vertexBuffer.size) - UInt64(instance.vertexOffset)
                * UInt64(MemoryLayout<PackedVertex>.size))
        geometry.vertexStride = MemoryLayout<PackedVertex>.size
        geometry.indexBuffer = MTL4BufferRangeMake(
            mesh.indexBuffer.getAddress()
                + UInt64(Int(instance.indexOffset[0]) * MemoryLayout<UInt32>.size),
            UInt64(Int(instance.indexCount[0]) * MemoryLayout<UInt32>.size))
        geometry.triangleCount = Int(instance.indexCount[0] / 3)
        geometry.indexType = .uint32
        geometry.opaque = true
        if !mesh.materials.isEmpty {
            geometry.opaque = mesh.materials[Int(instance.materialIndex)].alphaMode == 1
        }
        geometries.append(geometry)

        descriptor = MTL4PrimitiveAccelerationStructureDescriptor()
        descriptor.geometryDescriptors = geometries
        descriptor.usage = .preferFastIntersection

        let prebuildInfo = RendererData.device.accelerationStructureSizes(descriptor: descriptor)

        accelerationStructure = RendererData.device.makeAccelerationStructure(
            size: prebuildInfo.accelerationStructureSize)!
        scratchBuffer = Buffer(
            size: prebuildInfo.buildScratchBufferSize, makeResidentNow: makeResidentNow)

        if makeResidentNow {
            RendererData.addResidentAllocation(accelerationStructure)
            allocated = true
        }
    }

    func makeResident() {
        guard !allocated else { return }
        scratchBuffer.makeResident()
        RendererData.addResidentAllocation(accelerationStructure)
        allocated = true
    }

    deinit {
        if allocated {
            RendererData.removeResidentAllocation(accelerationStructure)
        }
    }

    func setName(name: String) {
        accelerationStructure.label = name
    }

    func destroyScratch() {
        scratchBuffer = nil
    }

    // TODO: Compaction
}
