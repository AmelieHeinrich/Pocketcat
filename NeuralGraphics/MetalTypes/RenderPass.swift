//
//  RenderPass.swift
//  Neural Graphics
//
//  Created by Amélie Heinrich on 25/02/2026.
//

import Metal
import simd

struct RenderPassDescriptor {
    var name: String = "Render Pass"
    var colorTextures: [MTLTexture] = []
    var clearColors: [simd_float4] = []
    var shouldClear: [Bool] = []
    
    var depthTexture: MTLTexture! = nil
    var shouldClearDepth: Bool = true
    var shouldStoreDepth: Bool = true
    
    mutating func setName(name: String) {
        self.name = name
    }
    
    mutating func addAttachment(texture: Texture, clearColor: simd_float4 = simd_float4(0.0, 0.0, 0.0, 1.0), shouldClear: Bool = true){
        self.colorTextures.append(texture.texture)
        self.clearColors.append(clearColor)
        self.shouldClear.append(shouldClear)
    }
    
    mutating func addAttachment(texture: MTLTexture, clearColor: simd_float4 = simd_float4(0.0, 0.0, 0.0, 1.0), shouldClear: Bool = true) {
        self.colorTextures.append(texture)
        self.clearColors.append(clearColor)
        self.shouldClear.append(shouldClear)
    }
    
    mutating func setDepthAttachment(texture: Texture, shouldClear: Bool = true, shouldStore: Bool = true) {
        self.depthTexture = texture.texture
        self.shouldClearDepth = shouldClear
        self.shouldStoreDepth = shouldStore
    }
}

class RenderPass {
    var descriptor: RenderPassDescriptor
    var encoder: MTL4RenderCommandEncoder
    
    init(descriptor: RenderPassDescriptor, cmdBuffer: MTL4CommandBuffer) {
        self.descriptor = descriptor
        
        let rpd = MTL4RenderPassDescriptor()
        for i in 0..<descriptor.colorTextures.count {
            rpd.colorAttachments[i].texture = descriptor.colorTextures[i]
            rpd.colorAttachments[i].clearColor = MTLClearColorMake(Double(descriptor.clearColors[i].x), Double(descriptor.clearColors[i].y), Double(descriptor.clearColors[i].z), Double(descriptor.clearColors[i].w))
            rpd.colorAttachments[i].storeAction = .store
            rpd.colorAttachments[i].loadAction = descriptor.shouldClear[i] ? .clear : .load
        }
        if descriptor.depthTexture != nil {
            rpd.depthAttachment.texture = descriptor.depthTexture
            rpd.depthAttachment.loadAction = descriptor.shouldClearDepth ? .clear : .load
            rpd.depthAttachment.clearDepth = 1.0
            rpd.depthAttachment.storeAction = descriptor.shouldStoreDepth ? .store : .dontCare
        }
        
        self.encoder = cmdBuffer.makeRenderCommandEncoder(descriptor: rpd)!
        self.encoder.label = descriptor.name
    }
    
    func end() {
        self.encoder.endEncoding()
    }
    
    func setPipeline(pipeline: RenderPipeline) {
        self.encoder.setRenderPipelineState(pipeline.pipelineState)
        if pipeline.depthStencilState != nil {
            self.encoder.setDepthStencilState(pipeline.depthStencilState)
        }
        self.encoder.setArgumentTable(RendererData.vertexTable, stages: .vertex)
        self.encoder.setArgumentTable(RendererData.fragmentTable, stages: .fragment)
    }
    
    func draw(primitiveType: MTLPrimitiveType, vertexCount: Int, vertexOffset: Int) {
        self.encoder.drawPrimitives(primitiveType: primitiveType, vertexStart: vertexOffset, vertexCount: vertexCount)
    }
    
    func drawIndexed(primitimeType: MTLPrimitiveType, buffer: Buffer, indexCount: Int, indexOffset: UInt64) {
        self.encoder.drawIndexedPrimitives(primitiveType: primitimeType, indexCount: indexCount, indexType: .uint32, indexBuffer: buffer.getAddress() + indexOffset, indexBufferLength: indexCount * MemoryLayout<UInt32>.size)
    }
    
    func setTexture(texture: Texture, index: Int, stages: MTLRenderStages) {
        if stages.contains(.vertex) {
            RendererData.vertexTable.setTexture(texture.texture.gpuResourceID, index: index)
        }
        if stages.contains(.fragment) {
            RendererData.fragmentTable.setTexture(texture.texture.gpuResourceID, index: index)
        }
    }
    
    func setTexture(texture: MTLTexture, index: Int, stages: MTLRenderStages) {
        if stages.contains(.vertex) {
            RendererData.vertexTable.setTexture(texture.gpuResourceID, index: index)
        }
        if stages.contains(.fragment) {
            RendererData.fragmentTable.setTexture(texture.gpuResourceID, index: index)
        }
    }
    
    func pushMarker(name: String) {
        self.encoder.pushDebugGroup(name)
    }
    
    func popMarker() {
        self.encoder.popDebugGroup()
    }
}
