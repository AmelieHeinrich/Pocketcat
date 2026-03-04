//
//  MeshPipeline.swift
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 03/03/2026.
//

import Metal

struct MeshPipelineDescriptor {
    var name: String = ""
    
    var objectFunction: String? = nil
    var meshFunction: String = ""
    var fragmentFunction: String? = nil
    
    var blendingEnabled: Bool = false
    var pixelFormats: [MTLPixelFormat] = []
    
    var depthFormat: MTLPixelFormat = .invalid
    var depthEnabled: Bool = false
    var depthWriteEnabled: Bool = false
    var depthCompareOp: MTLCompareFunction = .less
    var primitiveTopologyClass: MTLPrimitiveTopologyClass = .triangle
    
    var supportsIndirect: Bool = false
}

class MeshPipeline {
    var descriptor: MeshPipelineDescriptor
    var pipelineState: MTLRenderPipelineState
    var depthStencilState: MTLDepthStencilState!
    
    init(descriptor: MeshPipelineDescriptor) {
        self.descriptor = descriptor
        
        let meshFunctionDescriptor = MTL4LibraryFunctionDescriptor()
        meshFunctionDescriptor.library = RendererData.library
        meshFunctionDescriptor.name = descriptor.meshFunction
        
        let fragmentFunctionDescriptor = MTL4LibraryFunctionDescriptor()
        if descriptor.fragmentFunction != nil {
            fragmentFunctionDescriptor.library = RendererData.library
            fragmentFunctionDescriptor.name = descriptor.fragmentFunction
        }
        
        let objectFunctionDescriptor = MTL4LibraryFunctionDescriptor()
        if descriptor.fragmentFunction != nil {
            objectFunctionDescriptor.library = RendererData.library
            objectFunctionDescriptor.name = descriptor.objectFunction
        }
        
        let pipelineDesc = MTL4MeshRenderPipelineDescriptor()
        pipelineDesc.label = descriptor.name
        pipelineDesc.meshFunctionDescriptor = meshFunctionDescriptor
        if descriptor.fragmentFunction != nil {
            pipelineDesc.fragmentFunctionDescriptor = fragmentFunctionDescriptor
        }
        if descriptor.objectFunction != nil {
            pipelineDesc.objectFunctionDescriptor = objectFunctionDescriptor
        }
        for i in 0..<descriptor.pixelFormats.count {
            pipelineDesc.colorAttachments[i].pixelFormat = descriptor.pixelFormats[i]
            if descriptor.blendingEnabled {
                pipelineDesc.colorAttachments[i].blendingState = .enabled;
                pipelineDesc.colorAttachments[i].rgbBlendOperation = .add;
                pipelineDesc.colorAttachments[i].alphaBlendOperation = .add;
                pipelineDesc.colorAttachments[i].sourceRGBBlendFactor = .sourceAlpha;
                pipelineDesc.colorAttachments[i].sourceAlphaBlendFactor = .sourceAlpha;
                pipelineDesc.colorAttachments[i].destinationRGBBlendFactor = .oneMinusSourceAlpha;
                pipelineDesc.colorAttachments[i].destinationAlphaBlendFactor = .oneMinusSourceAlpha;
            }
        }
        
        if descriptor.depthEnabled {
            let depthDescriptor = MTLDepthStencilDescriptor()
            depthDescriptor.depthCompareFunction = descriptor.depthCompareOp
            depthDescriptor.isDepthWriteEnabled = descriptor.depthWriteEnabled

            self.depthStencilState = RendererData.device.makeDepthStencilState(descriptor: depthDescriptor)!
        }
        
        self.pipelineState = try! RendererData.compiler.makeRenderPipelineState(descriptor: pipelineDesc)
    }
}
