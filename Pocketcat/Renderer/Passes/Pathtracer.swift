//
//  PrimaryRayTestPass.swift
//  Pocketcat
//
//  Created by Amélie Heinrich on 20/03/2026.
//

import Metal
import simd

class Pathtracer: Pass {
    private let pipeline: ComputePipeline
    private var ift: MTLIntersectionFunctionTable
    private var colorTexture: Texture

    override init() {
        pipeline = ComputePipeline(function: "pathtracer", linkedFunctions: ["alpha_any_hit"])
        ift = pipeline.createIFT()

        let colorDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 1, height: 1, mipmapped: false)
        colorDesc.usage = [.shaderRead, .renderTarget, .shaderWrite]
        let color = Texture(descriptor: colorDesc)
        color.setLabel(name: "PT Texture")
        self.colorTexture = color

        super.init()
    }

    override func resize(renderWidth: Int, renderHeight: Int, outputWidth: Int, outputHeight: Int) {
        colorTexture.resize(width: renderWidth, height: renderHeight)
    }

    override func render(context: FrameContext) {
        guard (context.scene != nil) else { return }

        ift.setBuffer(context.sceneBuffer.buffer.buffer, offset: 0, index: 0)

        let cp = context.cmdBuffer.beginComputePass(name: "Pathtracer")
        cp.consumerBarrier(before: .dispatch, after: [.dispatch, .accelerationStructure])
        cp.setPipeline(pipeline: pipeline)
        cp.setBuffer(buf: context.sceneBuffer.buffer, index: 0)
        cp.setTexture(texture: self.colorTexture, index: 0)
        cp.setIFT(ift, index: 1)
        cp.dispatch(threads: MTLSizeMake((colorTexture.texture.width + 7) / 8, (colorTexture.texture.height + 7) / 8, 1), threadsPerGroup: MTLSizeMake(8, 8, 1))
        cp.end()

        context.resources.register(colorTexture, for: "HDR")
    }
}
