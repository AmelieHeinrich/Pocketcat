//
//  Tonemap.swift
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 02/03/2026.
//

import Metal
import simd
internal import QuartzCore

class TonemapPass: Pass {
    private let pipeline: RenderPipeline
    private unowned let settings: RendererSettings

    init(settings: RendererSettings) {
        var pipelineDesc = RenderPipelineDescriptor()
        pipelineDesc.name = "Tonemap"
        pipelineDesc.vertexFunction = "tonemap_vs"
        pipelineDesc.fragmentFunction = "tonemap_fs"
        pipelineDesc.pixelFormats = [.bgra8Unorm]

        self.pipeline = RenderPipeline(descriptor: pipelineDesc)
        self.settings = settings

        super.init()
    }

    override func render(context: FrameContext) {
        let forward = context.resources.get("Forward.Color") as Texture?
        guard let forward = forward else { return }

        var rpDesc = RenderPassDescriptor()
        rpDesc.setName(name: "Tonemap")
        rpDesc.addAttachment(texture: context.drawable.texture, shouldClear: false)

        let rp = context.cmdBuffer.beginRenderPass(descriptor: rpDesc)

        var gamma = settings.tonemapGamma
        rp.consumerBarrier(before: .fragment, after: .fragment)

        rp.setPipeline(pipeline: self.pipeline)
        rp.setTexture(texture: forward, index: 0, stages: .fragment)
        rp.setBytes(allocator: context.allocator, index: 0, bytes: &gamma, size: MemoryLayout<Float>.size, stages: .fragment)
        rp.draw(primitiveType: .triangle, vertexCount: 3, vertexOffset: 0)

        rp.end()
    }
}
