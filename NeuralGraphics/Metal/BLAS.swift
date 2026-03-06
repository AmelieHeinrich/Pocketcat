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

    init(model: Mesh, makeResidentNow: Bool = true) {
        for mesh in model.instances {
            let geometry = MTL4AccelerationStructureTriangleGeometryDescriptor()
            geometry.vertexBuffer = MTL4BufferRangeMake(
                model.vertexBuffer.getAddress(), UInt64(model.vertexBuffer.size))
            geometry.vertexStride = MemoryLayout<PackedVertex>.size
            geometry.indexBuffer = MTL4BufferRangeMake(
                model.indexBuffer.getAddress()
                    + UInt64(Int(mesh.indexOffset[0]) * MemoryLayout<Int>.size),
                UInt64(Int(mesh.indexCount[0]) * MemoryLayout<Int>.size))
            geometry.triangleCount = Int(mesh.indexCount[0] / 3)
            geometry.indexType = .uint32
            geometry.opaque = true

            geometries.append(geometry)
        }

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
