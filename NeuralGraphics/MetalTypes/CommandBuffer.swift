//
//  CommandBuffer.swift
//  Neural Graphics
//
//  Created by Amélie Heinrich on 25/02/2026.
//

import Metal

class CommandBuffer {
    var commandBuffer: MTL4CommandBuffer
    var allocator: MTL4CommandAllocator
    var lastSignaledValue: UInt64 = 0

    init() {
        self.commandBuffer = RendererData.device.makeCommandBuffer()!
        self.allocator = RendererData.device.makeCommandAllocator()!
    }
    
    func setName(name: String) {
        self.commandBuffer.label = name
    }
    
    func begin() {
        self.allocator.reset()
        self.commandBuffer.beginCommandBuffer(allocator: self.allocator)
        RendererData.residencySet.commit()
    }
    
    func end() {
        self.commandBuffer.endCommandBuffer()
    }
    
    func beginRenderPass(descriptor: RenderPassDescriptor) -> RenderPass {
        return RenderPass(descriptor: descriptor, cmdBuffer: self.commandBuffer)
    }
    
    func beginMLPass(name: String) -> MLPass {
        return MLPass(name: name, cmdBuffer: self.commandBuffer)
    }
    
    func commit() {
        RendererData.cmdQueue.commit([commandBuffer])
    }
    
    func pushMarker(name: String) {
        self.commandBuffer.pushDebugGroup(name)
    }
    
    func popMarker() {
        self.commandBuffer.popDebugGroup()
    }
}
