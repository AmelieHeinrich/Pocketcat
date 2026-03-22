//
//  AccumulationDenoiserPass.swift
//  Pocketcat
//
//  Created by Amélie Heinrich on 22/03/2026.
//

import Metal

class AccumulationDenoiserPass: Pass {
    private let pipe: ComputePipeline
    private var accumTextures: [Texture] = []
    private var currentIdx: Int = 0

    override init() {
        pipe = ComputePipeline(function: "accumulation_denoiser", name: "Accumulation Denoiser")
        super.init()
    }

    private func makeAccumTexture(width: Int, height: Int) -> Texture {
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false)
        desc.usage = [.shaderRead, .shaderWrite]
        let tex = Texture(descriptor: desc)
        tex.setLabel(name: "PT Accumulation")
        return tex
    }

    override func resize(renderWidth: Int, renderHeight: Int, outputWidth: Int, outputHeight: Int) {
        accumTextures = [
            makeAccumTexture(width: renderWidth, height: renderHeight),
            makeAccumTexture(width: renderWidth, height: renderHeight)
        ]
        currentIdx = 0
    }

    override func render(context: FrameContext) {
        let rawSample  = context.resources.get("PT.RawSample")          as Texture?
        let motionVecs = context.resources.get("GBuffer.MotionVectors")  as Texture?

        guard let rawSample = rawSample, let motionVecs = motionVecs else { return }
        guard accumTextures.count == 2 else { return }

        let prevIdx = currentIdx
        let nextIdx = 1 - currentIdx

        let w = accumTextures[nextIdx].texture.width
        let h = accumTextures[nextIdx].texture.height

        let cp = context.cmdBuffer.beginComputePass(name: "AccumulationDenoiser")
        cp.consumerBarrier(before: .dispatch, after: [.dispatch, .fragment])
        cp.setPipeline(pipeline: pipe)
        cp.setTexture(texture: rawSample,            index: 0)
        cp.setTexture(texture: motionVecs,           index: 1)
        cp.setTexture(texture: accumTextures[prevIdx], index: 2)
        cp.setTexture(texture: accumTextures[nextIdx], index: 3)
        cp.dispatch(threads: MTLSizeMake((w + 7) / 8, (h + 7) / 8, 1), threadsPerGroup: MTLSizeMake(8, 8, 1))
        cp.end()

        currentIdx = nextIdx
        context.resources.register(accumTextures[currentIdx], for: "HDR")
    }
}
