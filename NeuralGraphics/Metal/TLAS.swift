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
    var indirectDescriptor: MTL4IndirectInstanceAccelerationStructureDescriptor
    var instanceDescriptors: [MTLIndirectAccelerationStructureInstanceDescriptor] = []
    var instanceBuffer: Buffer
    var instanceCountBuffer: Buffer
    var scratchBuffer: Buffer! = nil
    var blasMap: [UInt64] = []
    private var allocated: Bool = false

    init(makeResidentNow: Bool = true) {
        instanceCountBuffer = Buffer(
            size: MemoryLayout<UInt32>.size, makeResidentNow: makeResidentNow)
        instanceCountBuffer.setName(name: "TLAS Instance Count Buffer")

        instanceBuffer = Buffer(
            size: MemoryLayout<MTLIndirectAccelerationStructureInstanceDescriptor>.size * 65536,
            makeResidentNow: makeResidentNow)
        instanceBuffer.setName(name: "TLAS Instance Buffer")

        descriptor = MTL4InstanceAccelerationStructureDescriptor()
        descriptor.instanceCount = 65536
        descriptor.instanceDescriptorType = .indirect
        descriptor.instanceDescriptorBuffer = MTL4BufferRangeMake(instanceBuffer.getAddress(), UInt64(instanceBuffer.size))
        descriptor.instanceDescriptorStride = MemoryLayout<MTLIndirectAccelerationStructureInstanceDescriptor>.stride
        descriptor.instanceTransformationMatrixLayout = .columnMajor

        indirectDescriptor = MTL4IndirectInstanceAccelerationStructureDescriptor()
        indirectDescriptor.maxInstanceCount = 65536
        indirectDescriptor.instanceCountBuffer = MTL4BufferRangeMake(instanceCountBuffer.getAddress(), UInt64(MemoryLayout<UInt32>.size))
        indirectDescriptor.instanceDescriptorType = .indirect
        indirectDescriptor.instanceDescriptorBuffer = MTL4BufferRangeMake(instanceBuffer.getAddress(), UInt64(instanceBuffer.size))
        indirectDescriptor.instanceDescriptorStride = MemoryLayout<MTLIndirectAccelerationStructureInstanceDescriptor>.stride
        indirectDescriptor.instanceTransformationMatrixLayout = .columnMajor

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
        instanceCountBuffer.makeResident()
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

    func addInstance(
        blas: BLAS, matrix: simd_float4x4 = simd_float4x4.identity, opaque: Bool = true
    ) {
        var instanceDescriptor = MTLIndirectAccelerationStructureInstanceDescriptor()
        instanceDescriptor.options = opaque ? .opaque : .nonOpaque
        instanceDescriptor.mask = 0xFF
        instanceDescriptor.accelerationStructureID = blas.accelerationStructure.gpuResourceID
        instanceDescriptor.intersectionFunctionTableOffset = 0
        instanceDescriptor.transformationMatrix.columns.0.x = matrix.columns.0.x
        instanceDescriptor.transformationMatrix.columns.0.y = matrix.columns.0.y
        instanceDescriptor.transformationMatrix.columns.0.z = matrix.columns.0.z
        instanceDescriptor.transformationMatrix.columns.1.x = matrix.columns.1.x
        instanceDescriptor.transformationMatrix.columns.1.y = matrix.columns.1.y
        instanceDescriptor.transformationMatrix.columns.1.z = matrix.columns.1.z
        instanceDescriptor.transformationMatrix.columns.2.x = matrix.columns.2.x
        instanceDescriptor.transformationMatrix.columns.2.y = matrix.columns.2.y
        instanceDescriptor.transformationMatrix.columns.2.z = matrix.columns.2.z
        instanceDescriptor.transformationMatrix.columns.3.x = matrix.columns.3.x
        instanceDescriptor.transformationMatrix.columns.3.y = matrix.columns.3.y
        instanceDescriptor.transformationMatrix.columns.3.z = matrix.columns.3.z

        instanceDescriptors.append(instanceDescriptor)
    }

    func update() {
        instanceDescriptors.withUnsafeBytes { bytes in
            instanceBuffer.write(bytes: bytes.baseAddress!, size: bytes.count)
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
