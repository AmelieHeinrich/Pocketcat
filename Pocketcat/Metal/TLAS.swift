//
//  TLAS.swift
//  Pocketcat
//
//  Created by Amélie Heinrich on 02/03/2026.
//

import Metal
import simd

let FRAMES_IN_FLIGHT = 3

class TLAS {
    var tlas: MTLAccelerationStructure
    var descriptor: MTL4InstanceAccelerationStructureDescriptor
    var indirectDescriptor: MTL4IndirectInstanceAccelerationStructureDescriptor
    var instanceDescriptors: [MTLIndirectAccelerationStructureInstanceDescriptor] = []
    var instanceBuffers: [Buffer] = []
    var instanceCountBuffers: [Buffer] = []
    var scratchBuffer: Buffer! = nil
    private var allocated: Bool = false

    init(makeResidentNow: Bool = true) {
        for i in 0..<FRAMES_IN_FLIGHT {
            let countBuf = Buffer(size: MemoryLayout<UInt32>.size, makeResidentNow: makeResidentNow)
            countBuf.setName(name: "TLAS Instance Count Buffer [\(i)]")
            instanceCountBuffers.append(countBuf)

            let instBuf = Buffer(
                size: MemoryLayout<MTLIndirectAccelerationStructureInstanceDescriptor>.size * 65536,
                makeResidentNow: makeResidentNow)
            instBuf.setName(name: "TLAS Instance Buffer [\(i)]")
            instanceBuffers.append(instBuf)
        }

        descriptor = MTL4InstanceAccelerationStructureDescriptor()
        descriptor.instanceCount = 65536
        descriptor.instanceDescriptorType = .indirect
        descriptor.instanceDescriptorBuffer = MTL4BufferRangeMake(
            instanceBuffers[0].getAddress(), UInt64(instanceBuffers[0].size))
        descriptor.instanceDescriptorStride =
            MemoryLayout<MTLIndirectAccelerationStructureInstanceDescriptor>.stride
        descriptor.instanceTransformationMatrixLayout = .columnMajor

        // indirectDescriptor is updated per-frame in updateFrameDescriptor()
        indirectDescriptor = MTL4IndirectInstanceAccelerationStructureDescriptor()
        indirectDescriptor.maxInstanceCount = 65536
        indirectDescriptor.instanceDescriptorType = .indirect
        indirectDescriptor.instanceDescriptorStride =
            MemoryLayout<MTLIndirectAccelerationStructureInstanceDescriptor>.stride
        indirectDescriptor.instanceTransformationMatrixLayout = .columnMajor

        let sizes = RendererData.device.accelerationStructureSizes(descriptor: descriptor)
        tlas = RendererData.device.makeAccelerationStructure(size: sizes.accelerationStructureSize)!
        scratchBuffer = Buffer(size: sizes.buildScratchBufferSize, makeResidentNow: makeResidentNow)

        if makeResidentNow {
            RendererData.addResidentAllocation(tlas)
            allocated = true
        }
    }

    // Call this at the start of gpuBuild, before encoding anything
    func updateFrameDescriptor(frameIdx: Int) {
        indirectDescriptor.instanceDescriptorBuffer = MTL4BufferRangeMake(
            instanceBuffers[frameIdx].getAddress(), UInt64(instanceBuffers[frameIdx].size))
        indirectDescriptor.instanceCountBuffer = MTL4BufferRangeMake(
            instanceCountBuffers[frameIdx].getAddress(), UInt64(instanceCountBuffers[frameIdx].size))
    }

    func makeResident() {
        guard !allocated else { return }
        for i in 0..<FRAMES_IN_FLIGHT {
            instanceBuffers[i].makeResident()
            instanceCountBuffers[i].makeResident()
        }
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
    }

    func addInstance(
        blas: BLAS, matrix: simd_float4x4 = simd_float4x4.identity, opaque: Bool = true, userID: UInt32 = 0
    ) {
        var instanceDescriptor = MTLIndirectAccelerationStructureInstanceDescriptor()
        instanceDescriptor.options = opaque ? .opaque : .nonOpaque
        instanceDescriptor.mask = 0xFF
        instanceDescriptor.accelerationStructureID = blas.accelerationStructure.gpuResourceID
        instanceDescriptor.intersectionFunctionTableOffset = 0
        instanceDescriptor.userID = userID
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

    func update(frameIdx: Int) {
        instanceDescriptors.withUnsafeBytes { bytes in
            instanceBuffers[frameIdx].write(bytes: bytes.baseAddress!, size: bytes.count)
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
