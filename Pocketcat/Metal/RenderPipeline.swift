//
//  RenderPipeline.swift
//  Pocketcat
//
//  Created by Amélie Heinrich on 25/02/2026.
//

import Metal

struct RenderPipelineDescriptor {
    var name: String = "Render Pipeline"

    var vertexFunction: String = ""
    var fragmentFunction: String? = nil

    var blendingEnabled: Bool = false
    var additiveBlending: Bool = false  // src=sourceAlpha, dst=one
    var pixelFormats: [MTLPixelFormat] = []

    var depthFormat: MTLPixelFormat = .invalid
    var depthEnabled: Bool = false
    var depthWriteEnabled: Bool = false
    var depthCompareOp: MTLCompareFunction = .less
    var primitiveTopologyClass: MTLPrimitiveTopologyClass = .triangle

    var linkedFunctions: [String] = []
    var supportsIndirect: Bool = false
}

class RenderPipeline {
    var descriptor: RenderPipelineDescriptor
    var pipelineState: MTLRenderPipelineState
    var depthStencilState: MTLDepthStencilState!
    var linkedFunctions: [MTLFunction] = []

    init(descriptor: RenderPipelineDescriptor) {
        self.descriptor = descriptor

        let vertexFn = RendererData.library.makeFunction(name: descriptor.vertexFunction)!
        let fragmentFn = descriptor.fragmentFunction.map {
            RendererData.library.makeFunction(name: $0)!
        }

        var mtlFunctions: [MTLFunction] = []
        for funcName in descriptor.linkedFunctions {
            mtlFunctions.append(RendererData.library.makeFunction(name: funcName)!)
        }

        let pipelineDesc = MTLRenderPipelineDescriptor()
        pipelineDesc.label = descriptor.name
        pipelineDesc.vertexFunction = vertexFn
        pipelineDesc.fragmentFunction = fragmentFn
        for i in 0..<descriptor.pixelFormats.count {
            pipelineDesc.colorAttachments[i].pixelFormat = descriptor.pixelFormats[i]
            if descriptor.blendingEnabled {
                pipelineDesc.colorAttachments[i].isBlendingEnabled = true
                pipelineDesc.colorAttachments[i].rgbBlendOperation = .add
                pipelineDesc.colorAttachments[i].alphaBlendOperation = .add
                pipelineDesc.colorAttachments[i].sourceRGBBlendFactor = .sourceAlpha
                pipelineDesc.colorAttachments[i].sourceAlphaBlendFactor = .sourceAlpha
                pipelineDesc.colorAttachments[i].destinationRGBBlendFactor = .oneMinusSourceAlpha
                pipelineDesc.colorAttachments[i].destinationAlphaBlendFactor = .oneMinusSourceAlpha
            } else if descriptor.additiveBlending {
                pipelineDesc.colorAttachments[i].isBlendingEnabled = true
                pipelineDesc.colorAttachments[i].rgbBlendOperation = .add
                pipelineDesc.colorAttachments[i].alphaBlendOperation = .add
                pipelineDesc.colorAttachments[i].sourceRGBBlendFactor = .sourceAlpha
                pipelineDesc.colorAttachments[i].sourceAlphaBlendFactor = .one
                pipelineDesc.colorAttachments[i].destinationRGBBlendFactor = .one
                pipelineDesc.colorAttachments[i].destinationAlphaBlendFactor = .one
            }
        }
        if descriptor.depthEnabled {
            pipelineDesc.depthAttachmentPixelFormat = descriptor.depthFormat
        }
        if !mtlFunctions.isEmpty {
            let linked = MTLLinkedFunctions()
            linked.functions = mtlFunctions
            pipelineDesc.fragmentLinkedFunctions = linked
        }
        if descriptor.supportsIndirect {
            pipelineDesc.supportIndirectCommandBuffers = true
        }
        pipelineDesc.inputPrimitiveTopology = descriptor.primitiveTopologyClass

        if descriptor.depthEnabled {
            let depthDescriptor = MTLDepthStencilDescriptor()
            depthDescriptor.depthCompareFunction = descriptor.depthCompareOp
            depthDescriptor.isDepthWriteEnabled = descriptor.depthWriteEnabled
            self.depthStencilState = RendererData.device.makeDepthStencilState(
                descriptor: depthDescriptor)!
        }

        self.pipelineState = try! RendererData.device.makeRenderPipelineState(descriptor: pipelineDesc)
        self.linkedFunctions = mtlFunctions
    }

    func createIFT() -> MTLIntersectionFunctionTable {
        let iftDesc = MTLIntersectionFunctionTableDescriptor()
        iftDesc.functionCount = linkedFunctions.count
        let ift = pipelineState.makeIntersectionFunctionTable(descriptor: iftDesc, stage: .fragment)!
        for (i, fn) in linkedFunctions.enumerated() {
            let handle = pipelineState.functionHandle(function: fn, stage: .fragment)!
            ift.setFunction(handle, index: i)
        }
        return ift
    }
}
