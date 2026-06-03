//
//  RTShadows.swift
//  Pocketcat
//
//  Created by Amélie Heinrich on 23/03/2026.
//

import Metal
import simd

struct RTShadowParameters {
    var frameIndex: UInt32 = 0
    var spp: UInt32 = 0
}

class RTShadows: Pass {
    private let tracePipeline: ComputePipeline
    private var ift: MTLIntersectionFunctionTable
    private var shadowMask: Texture
    private var accumulationFrame: UInt32 = 0
    private unowned var settings: SettingsRegistry

    init(settings: SettingsRegistry) {
        self.settings = settings
        self.settings.register(
            int: "RTShadows.SamplesPerPixel", label: "Samples per pixel", default: 1, range: 1...32)

        tracePipeline = ComputePipeline(function: "rt_shadows", linkedFunctions: ["alpha_any_hit"])
        ift = tracePipeline.createIFT()

        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r8Unorm, width: 1, height: 1, mipmapped: false)
        desc.usage = [.shaderRead, .shaderWrite]
        shadowMask = Texture(descriptor: desc)
        shadowMask.setLabel(name: "RTShadows.NoisyMask")

        super.init()
    }

    override func resize(renderWidth: Int, renderHeight: Int, outputWidth: Int, outputHeight: Int) {
        shadowMask.resize(width: renderWidth, height: renderHeight)
        accumulationFrame = 0
    }

    override func render(context: FrameContext) {
        guard context.scene != nil else { return }

        let cp = context.cmdBuffer.beginComputePass(name: "RT Shadows")
        recordTrace(context: context, cp: cp)
        cp.end()

        context.resources.register(shadowMask, for: "RTShadows.Output")
        context.resources.addVisualizer(texture: shadowMask, label: "RTShadows.Raw",
            fragmentFunction: "texviz_single_channel_fs")

        accumulationFrame &+= 1
    }

    private func recordTrace(context: FrameContext, cp: ComputePass) {
        let w = shadowMask.texture.width
        let h = shadowMask.texture.height

        let depth = context.resources.get("GBuffer.Depth") as Texture?
        let normals = context.resources.get("GBuffer.Normal") as Texture?
        guard let depth = depth, let normals = normals else { return }

        var parameters = RTShadowParameters()
        parameters.spp = UInt32(settings.int("RTShadows.SamplesPerPixel", default: 1))
        parameters.frameIndex = accumulationFrame

        ift.setBuffer(context.sceneBuffer.buffer.buffer, offset: 0, index: 0)

        cp.consumerBarrier(before: .dispatch, after: [.dispatch, .accelerationStructure, .fragment])
        cp.setPipeline(pipeline: tracePipeline)
        cp.setBuffer(buf: context.sceneBuffer.buffer, index: 0)
        cp.setBytes(allocator: context.allocator, index: 1, bytes: &parameters, size: MemoryLayout<RTShadowParameters>.size)
        cp.setIFT(ift, index: 2)
        cp.setTexture(texture: shadowMask, index: 0)
        cp.setTexture(texture: depth, index: 1)
        cp.setTexture(texture: normals, index: 2)
        cp.dispatch(
            threads: MTLSizeMake((w + 7) / 8, (h + 7) / 8, 1), threadsPerGroup: MTLSizeMake(8, 8, 1)
        )
    }
}
