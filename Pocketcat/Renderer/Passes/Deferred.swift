//
//  Deferred.swift
//  Pocketcat
//
//  Created by Amélie Heinrich on 21/03/2026.
//

import Metal
import simd

class DeferredPass: Pass {
    private let pipeline: ComputePipeline
    private var ift: MTLIntersectionFunctionTable
    private var colorTexture: Texture

    override init() {
        pipeline = ComputePipeline(function: "deferred_kernel", linkedFunctions: ["alpha_any_hit"])
        ift = pipeline.createIFT()

        let colorDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 1, height: 1, mipmapped: false)
        colorDesc.usage = [.shaderRead, .renderTarget, .shaderWrite]
        let color = Texture(descriptor: colorDesc)
        color.setLabel(name: "HDR Lighting")
        self.colorTexture = color

        super.init()
    }

    override func resize(renderWidth: Int, renderHeight: Int, outputWidth: Int, outputHeight: Int) {
        colorTexture.resize(width: renderWidth, height: renderHeight)
    }

    override func render(context: FrameContext) {
        let depth = context.resources.get("GBuffer.Depth") as Texture?
        let albedo = context.resources.get("GBuffer.Albedo") as Texture?
        let normal = context.resources.get("GBuffer.Normal") as Texture?
        let orm = context.resources.get("GBuffer.ORM") as Texture?
        let emissive = context.resources.get("GBuffer.Emissive") as Texture?
        
        guard let depth = depth else { return }
        guard let albedo = albedo else { return }
        guard let normal = normal else { return }
        guard let orm = orm else { return }
        guard let emissive = emissive else { return }
        guard (context.scene != nil) else { return }

        ift.setBuffer(context.sceneBuffer.buffer.buffer, offset: 0, index: 0)

        let cp = context.cmdBuffer.beginComputePass(name: "Deferred")
        cp.consumerBarrier(before: .dispatch, after: [.dispatch, .accelerationStructure])
        cp.setPipeline(pipeline: pipeline)
        cp.setBuffer(buf: context.sceneBuffer.buffer, index: 0)
        cp.setIFT(ift, index: 1)
        cp.setTexture(texture: depth, index: 0)
        cp.setTexture(texture: albedo, index: 1)
        cp.setTexture(texture: normal, index: 2)
        cp.setTexture(texture: orm, index: 3)
        cp.setTexture(texture: emissive, index: 4)
        cp.setTexture(texture: self.colorTexture, index: 5)
        cp.dispatch(threads: MTLSizeMake((colorTexture.texture.width + 7) / 8, (colorTexture.texture.height + 7) / 8, 1), threadsPerGroup: MTLSizeMake(8, 8, 1))
        cp.end()

        context.resources.register(colorTexture, for: "HDR")
    }
}

