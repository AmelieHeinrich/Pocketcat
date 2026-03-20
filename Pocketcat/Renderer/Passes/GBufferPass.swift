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

        super.init()
    }

    override func resize(width: Int, height: Int) {
        albedoTexture.resize(width: width, height: height)
        normalTexture.resize(width: width, height: height)
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
        cp.dispatch(threads: MTLSizeMake(tgW, tgH, 1), threadsPerGroup: MTLSizeMake(8, 8, 1))
        cp.end()

        context.resources.register(albedoTexture, for: "Forward.Color")
        context.resources.register(normalTexture, for: "GBuffer.Normal")
    }
}
