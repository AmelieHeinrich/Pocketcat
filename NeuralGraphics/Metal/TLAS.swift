//
//  TLAS.swift
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 02/03/2026.
//

import Metal

class TLAS {
    var tlas: MTLAccelerationStructure
    var descriptor: MTL4InstanceAccelerationStructureDescriptor
    var instanceDescriptors: [MTLAccelerationStructureInstanceDescriptor] = []
    var instanceBuffer: Buffer
    var scratchBuffer: Buffer! = nil
    var blasMap: [UInt64] = []
    
    init() {
        instanceBuffer = Buffer(size: MemoryLayout<MTLAccelerationStructureInstanceDescriptor>.size * 16)
        instanceBuffer.setName(name: "TLAS Instance Buffer")
        
        descriptor = MTL4InstanceAccelerationStructureDescriptor()
        descriptor.instanceCount = 16
        descriptor.instanceDescriptorType = .default
        descriptor.instanceDescriptorBuffer = MTL4BufferRangeMake(instanceBuffer.getAddress(), UInt64(instanceBuffer.size))
        
        let sizes = RendererData.device.accelerationStructureSizes(descriptor: descriptor)
        tlas = RendererData.device.makeAccelerationStructure(size: sizes.accelerationStructureSize)!
        scratchBuffer = Buffer(size: sizes.buildScratchBufferSize)
        
        RendererData.residencySet.addAllocation(tlas)
    }
    
    deinit {
        RendererData.residencySet.removeAllocation(tlas)
    }
    
    func resetInstanceBuffer() {
        instanceDescriptors.removeAll()
        blasMap.removeAll()
    }
    
    func addInstance(blas: BLAS) {
        var found = -1
        for i in 0..<blasMap.count {
            if blasMap[i] == blas.accelerationStructure.gpuResourceID._impl {
                found = i
                break
            }
        }
        if found == -1 {
            blasMap.append(blas.accelerationStructure.gpuResourceID._impl)
            found = blasMap.count - 1
        }
        
        var instanceDescriptor = MTLAccelerationStructureInstanceDescriptor()
        instanceDescriptor.options = .nonOpaque
        instanceDescriptor.mask = 0xFF
        instanceDescriptor.accelerationStructureIndex = UInt32(found)
        instanceDescriptor.transformationMatrix.columns.0[0] = 1.0
        instanceDescriptor.transformationMatrix.columns.1[1] = 1.0
        instanceDescriptor.transformationMatrix.columns.2[2] = 1.0
        
        instanceDescriptors.append(instanceDescriptor)
    }
    
    func update() {
        instanceDescriptors.withUnsafeBytes() { bytes in
            instanceBuffer.write(bytes: bytes.baseAddress!, size: MemoryLayout<MTLAccelerationStructureInstanceDescriptor>.size * instanceDescriptors.count)
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
