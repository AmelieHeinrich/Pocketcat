//
//  ICB.swift
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 08/03/2026.
//

import Metal

class ICB {
    var buffer: Buffer
    var cmdBuffer: MTLIndirectCommandBuffer
    
    init(inherit: Bool, commandTypes: MTLIndirectCommandType, maxCommandCount: Int) {
        let descriptor = MTLIndirectCommandBufferDescriptor()
        descriptor.commandTypes = commandTypes
        descriptor.inheritBuffers = inherit
        descriptor.inheritPipelineState = true
        
        self.cmdBuffer = RendererData.device.makeIndirectCommandBuffer(descriptor: descriptor, maxCommandCount: maxCommandCount)!
        self.cmdBuffer.label = "ICB"
        
        var resourceID = self.cmdBuffer.gpuResourceID
        self.buffer = Buffer(bytes: &resourceID, size: MemoryLayout<UInt64>.size)
        
        RendererData.addResidentAllocation(self.cmdBuffer)
    }
}
