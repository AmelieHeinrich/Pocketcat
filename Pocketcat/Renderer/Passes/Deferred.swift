//
//  Deferred.swift
//  Pocketcat
//
//  Created by Amélie Heinrich on 21/03/2026.
//

import Metal
import simd

private struct DeferredParameters {
    var depth: UInt64 = 0
    var albedo: UInt64 = 0
    var normal: UInt64 = 0
    var orm: UInt64 = 0
    var emissive: UInt64 = 0
    var mask: UInt64 = 0
    var ao: UInt64 = 0
    var output: UInt64 = 0
    var aoResolutionScale: Float = 1.0
    var aoEnabled: UInt32 = 0
}

class DeferredPass: Pass {
    private let pipeline: ComputePipeline
    private var colorTexture: Texture

    override init() {
        pipeline = ComputePipeline(function: "deferred_kernel", linkedFunctions: ["alpha_any_hit"])

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
        let mask = context.resources.get("RTShadows.Output") as Texture?
        let ao = context.resources.get("RTAO.Mask") as Texture?

        guard let depth = depth else { return }
        guard let albedo = albedo else { return }
        guard let normal = normal else { return }
        guard let orm = orm else { return }
        guard let emissive = emissive else { return }
        guard let mask = mask else { return }
        guard (context.scene != nil) else { return }

        var params = DeferredParameters()
        params.depth = depth.texture.gpuResourceID._impl
        params.albedo = albedo.texture.gpuResourceID._impl
        params.normal = normal.texture.gpuResourceID._impl
        params.orm = orm.texture.gpuResourceID._impl
        params.emissive = emissive.texture.gpuResourceID._impl
        params.mask = mask.texture.gpuResourceID._impl
        params.output = colorTexture.texture.gpuResourceID._impl

        if let ao = ao {
            params.ao = ao.texture.gpuResourceID._impl
            params.aoResolutionScale = Float(ao.texture.width) / Float(depth.texture.width)
            params.aoEnabled = 1
        }

        let cp = context.cmdBuffer.beginComputePass(name: "Deferred")
        cp.consumerBarrier(before: .dispatch, after: [.dispatch, .accelerationStructure, .blit])
        cp.setPipeline(pipeline: pipeline)
        cp.setBuffer(buf: context.sceneBuffer.buffer, index: 0)
        cp.setBytes(allocator: context.allocator, index: 2, bytes: &params, size: MemoryLayout<DeferredParameters>.size)
        cp.dispatch(threads: MTLSizeMake((colorTexture.texture.width + 7) / 8, (colorTexture.texture.height + 7) / 8, 1), threadsPerGroup: MTLSizeMake(8, 8, 1))
        cp.end()

        context.resources.register(colorTexture, for: "HDR")
    }
}
