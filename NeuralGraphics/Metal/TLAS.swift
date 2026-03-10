//
//  TLAS.swift
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 02/03/2026.
//

import Metal
import simd

class TLAS {
    var tlas: MTLAccelerationStructure
    var descriptor: MTL4InstanceAccelerationStructureDescriptor
    var instanceDescriptors: [MTLIndirectAccelerationStructureInstanceDescriptor] = []
    var instanceBuffer: Buffer
    var scratchBuffer: Buffer! = nil
    var blasMap: [UInt64] = []
    private var allocated: Bool = false

    init(makeResidentNow: Bool = true) {
        instanceBuffer = Buffer(
            size: MemoryLayout<MTLIndirectAccelerationStructureInstanceDescriptor>.size * 65536,
            makeResidentNow: makeResidentNow)
        instanceBuffer.setName(name: "TLAS Instance Buffer")

        descriptor = MTL4InstanceAccelerationStructureDescriptor()
        descriptor.instanceCount = 65536
        descriptor.instanceDescriptorType = .indirect
        descriptor.instanceDescriptorBuffer = MTL4BufferRangeMake(
            instanceBuffer.getAddress(), UInt64(instanceBuffer.size))
        descriptor.instanceDescriptorStride =
            MemoryLayout<MTLIndirectAccelerationStructureInstanceDescriptor>.size
        descriptor.instanceTransformationMatrixLayout = .columnMajor

        let sizes = RendererData.device.accelerationStructureSizes(descriptor: descriptor)
        tlas = RendererData.device.makeAccelerationStructure(size: sizes.accelerationStructureSize)!
        scratchBuffer = Buffer(size: sizes.buildScratchBufferSize, makeResidentNow: makeResidentNow)

        if makeResidentNow {
            RendererData.addResidentAllocation(tlas)
            allocated = true
        }
    }

    func makeResident() {
        guard !allocated else { return }
        instanceBuffer.makeResident()
        scratchBuffer.makeResident()
        RendererData.addResidentAllocation(tlas)
        allocated = true
    }

    deinit {
        if allocated {
            RendererData.removeResidentAllocation(tlas)
        }
    }

    func resetInstanceBuffer() {
        instanceDescriptors.removeAll()
        blasMap.removeAll()
    }

    func addInstance(blas: BLAS, matrix: simd_float4x4 = simd_float4x4.identity) {
        var instanceDescriptor = MTLIndirectAccelerationStructureInstanceDescriptor()
        instanceDescriptor.options = .opaque
        instanceDescriptor.mask = 0xFF
        instanceDescriptor.accelerationStructureID = blas.accelerationStructure.gpuResourceID
        for i in 0..<3 {
            instanceDescriptor.transformationMatrix.columns.0[Int32(i)] = matrix.columns.0[i]
            instanceDescriptor.transformationMatrix.columns.1[Int32(i)] = matrix.columns.1[i]
            instanceDescriptor.transformationMatrix.columns.2[Int32(i)] = matrix.columns.2[i]
            instanceDescriptor.transformationMatrix.columns.3[Int32(i)] = matrix.columns.3[i]
        }

        instanceDescriptors.append(instanceDescriptor)
    }

    func update() {
        instanceDescriptors.withUnsafeBytes { bytes in
            instanceBuffer.write(
                bytes: bytes.baseAddress!,
                size: bytes.count)
        }
    }

    func getResourceID() -> UInt64 {
        return tlas.gpuResourceID._impl
    }

    func setName(name: String) {
        tlas.label = name
    }

    func destroyScratch() {
        scratchBuffer = nil
    }
}
