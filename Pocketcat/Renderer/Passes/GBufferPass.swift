//
//  GBufferPass.swift
//  Pocketcat
//
//  Created by Amélie Heinrich on 20/03/2026.
//

import Metal
import simd

class GBufferPass: Pass {
    private let pipe: ComputePipeline
    private var albedoTexture: Texture
    private var normalTexture: Texture
    private var ormTexture: Texture
    private var emissiveTexture: Texture

    override init() {
        pipe = ComputePipeline(function: "generate_gbuffer", name: "Generate GBuffer")

        let albedoDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false)
        albedoDesc.usage = [.shaderRead, .shaderWrite]
        self.albedoTexture = Texture(descriptor: albedoDesc)
        self.albedoTexture.setLabel(name: "GBuffer Albedo")

        let normalDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 1, height: 1, mipmapped: false)
        normalDesc.usage = [.shaderRead, .shaderWrite]
        self.normalTexture = Texture(descriptor: normalDesc)
        self.normalTexture.setLabel(name: "GBuffer Normal")
        
        let ormDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg8Unorm, width: 1, height: 1, mipmapped: false)
        ormDesc.usage = [.shaderRead, .shaderWrite]
        self.ormTexture = Texture(descriptor: ormDesc)
        self.ormTexture.setLabel(name: "GBuffer ORM")
        
        let emissiveDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false)
        emissiveDesc.usage = [.shaderRead, .shaderWrite]
        self.emissiveTexture = Texture(descriptor: emissiveDesc)
        self.emissiveTexture.setLabel(name: "GBuffer Emissive")

        super.init()
    }

    override func resize(renderWidth: Int, renderHeight: Int, outputWidth: Int, outputHeight: Int) {
        albedoTexture.resize(width: renderWidth, height: renderHeight)
        normalTexture.resize(width: renderWidth, height: renderHeight)
        ormTexture.resize(width: renderWidth, height: renderHeight)
        emissiveTexture.resize(width: renderWidth, height: renderHeight)
    }

    override func render(context: FrameContext) {
        let visibility = context.resources.get("Visibility") as Texture?
        let depth = context.resources.get("GBuffer.Depth") as Texture?
        guard let visibility = visibility, let depth = depth else { return }

        let width = albedoTexture.texture.width
        let height = albedoTexture.texture.height
        let tgW = (width + 7) / 8
        let tgH = (height + 7) / 8

        let cp = context.cmdBuffer.beginComputePass(name: "GBuffer")
        cp.consumerBarrier(before: .dispatch, after: [.dispatch, .fragment])
        cp.setPipeline(pipeline: pipe)
        cp.setBuffer(buf: context.sceneBuffer.buffer, index: 0)
        cp.setTexture(texture: visibility, index: 0)
        cp.setTexture(texture: depth, index: 1)
        cp.setTexture(texture: albedoTexture, index: 2)
        cp.setTexture(texture: normalTexture, index: 3)
        cp.setTexture(texture: ormTexture, index: 4)
        cp.setTexture(texture: emissiveTexture, index: 5)
        cp.dispatch(threads: MTLSizeMake(tgW, tgH, 1), threadsPerGroup: MTLSizeMake(8, 8, 1))
        cp.end()

        context.resources.register(albedoTexture, for: "GBuffer.Albedo")
        context.resources.register(normalTexture, for: "GBuffer.Normal")
        context.resources.register(ormTexture, for: "GBuffer.ORM")
        context.resources.register(emissiveTexture, for: "GBuffer.Emissive")
    }
}
