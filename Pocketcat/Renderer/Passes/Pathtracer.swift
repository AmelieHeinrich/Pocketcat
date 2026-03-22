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
    private var rawSampleTexture: Texture
    private var accumulationFrame: UInt32 = 0

    override init() {
        pipeline = ComputePipeline(function: "pathtracer", linkedFunctions: ["alpha_any_hit"])
        ift = pipeline.createIFT()

        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 1, height: 1, mipmapped: false)
        desc.usage = [.shaderRead, .shaderWrite]
        let tex = Texture(descriptor: desc)
        tex.setLabel(name: "PT Raw Sample")
        self.rawSampleTexture = tex

        super.init()
    }

    override func resize(renderWidth: Int, renderHeight: Int, outputWidth: Int, outputHeight: Int) {
        rawSampleTexture.resize(width: renderWidth, height: renderHeight)
        accumulationFrame = 0
    }

    override func render(context: FrameContext) {
        guard context.scene != nil else { return }

        let depth    = context.resources.get("GBuffer.Depth")    as Texture?
        let albedo   = context.resources.get("GBuffer.Albedo")   as Texture?
        let normal   = context.resources.get("GBuffer.Normal")   as Texture?
        let orm      = context.resources.get("GBuffer.ORM")      as Texture?
        let emissive = context.resources.get("GBuffer.Emissive") as Texture?

        guard let depth = depth, let albedo = albedo,
              let normal = normal, let orm = orm, let emissive = emissive else { return }

        ift.setBuffer(context.sceneBuffer.buffer.buffer, offset: 0, index: 0)

        let w = rawSampleTexture.texture.width
        let h = rawSampleTexture.texture.height

        let cp = context.cmdBuffer.beginComputePass(name: "Pathtracer")
        cp.consumerBarrier(before: .dispatch, after: [.dispatch, .accelerationStructure, .fragment])
        cp.setPipeline(pipeline: pipeline)
        cp.setBuffer(buf: context.sceneBuffer.buffer, index: 0)
        cp.setIFT(ift, index: 1)
        var fi = accumulationFrame
        cp.setBytes(allocator: context.allocator, index: 2, bytes: &fi, size: MemoryLayout<UInt32>.size)
        cp.setTexture(texture: depth,    index: 0)
        cp.setTexture(texture: albedo,   index: 1)
        cp.setTexture(texture: normal,   index: 2)
        cp.setTexture(texture: orm,      index: 3)
        cp.setTexture(texture: emissive, index: 4)
        cp.setTexture(texture: rawSampleTexture, index: 5)
        cp.dispatch(threads: MTLSizeMake((w + 7) / 8, (h + 7) / 8, 1), threadsPerGroup: MTLSizeMake(8, 8, 1))
        cp.end()

        accumulationFrame &+= 1
        context.resources.register(rawSampleTexture, for: "PT.RawSample")
    }
}
