//
//  RenderPipeline.swift
//  Neural Graphics
//
//  Created by Amélie Heinrich on 25/02/2026.
//

import Metal

struct RenderPipelineDescriptor {
    var name: String = ""
    
    var vertexFunction: String = ""
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

class RenderPipeline {
    var descriptor: RenderPipelineDescriptor
    var pipelineState: MTLRenderPipelineState
    var depthStencilState: MTLDepthStencilState!
    
    init(descriptor: RenderPipelineDescriptor) {
        self.descriptor = descriptor
        
        let vertexFunctionDescriptor = MTL4LibraryFunctionDescriptor()
        vertexFunctionDescriptor.library = RendererData.library
        vertexFunctionDescriptor.name = descriptor.vertexFunction
        
        let fragmentFunctionDescriptor = MTL4LibraryFunctionDescriptor()
        fragmentFunctionDescriptor.library = RendererData.library
        if descriptor.fragmentFunction != nil {
            fragmentFunctionDescriptor.name = descriptor.fragmentFunction
        }
        
        let pipelineDesc = MTL4RenderPipelineDescriptor()
        pipelineDesc.label = descriptor.name
        pipelineDesc.vertexFunctionDescriptor = vertexFunctionDescriptor
        if descriptor.fragmentFunction != nil {
            pipelineDesc.fragmentFunctionDescriptor = fragmentFunctionDescriptor
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
