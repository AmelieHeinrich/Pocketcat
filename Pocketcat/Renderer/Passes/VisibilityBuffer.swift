//
//  VisibilityBuffer.swift
//  Pocketcat
//
//  Created by Amélie Heinrich on 20/03/2026.
//

import Metal
import simd

class VisibilityBufferPass: Pass {
    private let vertexPipe: RenderPipeline
    private let meshPipe: MeshPipeline
    private var visibilityTexture: Texture
    private var depthTexture: Texture
    private unowned let settings: RendererSettings

    init(settings: RendererSettings) {
        var pipelineDesc = RenderPipelineDescriptor()
        pipelineDesc.name = "Visibility (VS)"
        pipelineDesc.vertexFunction = "visibility_vs"
        pipelineDesc.fragmentFunction = "visibility_fs_vs"
        pipelineDesc.pixelFormats = [.rg32Uint]
        pipelineDesc.depthEnabled = true
        pipelineDesc.depthFormat = .depth32Float
        pipelineDesc.depthCompareOp = .less
        pipelineDesc.depthWriteEnabled = true
        pipelineDesc.supportsIndirect = true
        
        var meshPipelineDesc = MeshPipelineDescriptor()
        meshPipelineDesc.name = "Visibility (MS)"
        meshPipelineDesc.objectFunction = "visibility_os"
        meshPipelineDesc.meshFunction = "visibility_ms"
        meshPipelineDesc.fragmentFunction = "visibility_fs_ms"
        meshPipelineDesc.pixelFormats = [.rg32Uint]
        meshPipelineDesc.depthEnabled = true
        meshPipelineDesc.depthFormat = .depth32Float
        meshPipelineDesc.depthCompareOp = .less
        meshPipelineDesc.depthWriteEnabled = true
        meshPipelineDesc.supportsIndirect = true

        self.vertexPipe = RenderPipeline(descriptor: pipelineDesc)
        self.meshPipe = MeshPipeline(descriptor: meshPipelineDesc)
        
        let visDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg32Uint, width: 1, height: 1, mipmapped: false)
        visDesc.usage = [.shaderRead, .renderTarget, .shaderWrite]
        
        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float, width: 1, height: 1, mipmapped: false)
        depthDesc.usage = [.shaderRead, .renderTarget, .shaderWrite]
        
        self.visibilityTexture = Texture(descriptor: visDesc)
        self.depthTexture = Texture(descriptor: depthDesc)
        self.settings = settings
        
        super.init()
    }
    
    override func resize(width: Int, height: Int) {
        visibilityTexture.resize(width: width, height: height)
        depthTexture.resize(width: width, height: height)
    }

    override func render(context: FrameContext) {
        let icb = context.resources.get("MainViewICB") as ICB?
        guard let icb = icb else { return }

        var rpDesc = RenderPassDescriptor()
        rpDesc.setName(name: "Visibility Buffer")
        rpDesc.addAttachment(texture: self.visibilityTexture)
        rpDesc.setDepthAttachment(texture: self.depthTexture)

        let rp = context.cmdBuffer.beginRenderPass(descriptor: rpDesc)
        rp.consumerBarrier(before: [.vertex, .object], after: [.dispatch])
        if settings.useMeshShader {
            rp.setMeshPipeline(pipeline: meshPipe)
        } else {
            rp.setPipeline(pipeline: vertexPipe)
        }
        rp.executeIndirect(icb: icb, maxCommandCount: 65536)
        rp.end()
        
        context.resources.register(visibilityTexture, for: "Visibility")
        context.resources.register(depthTexture, for: "GBuffer.Depth")
    }
}
