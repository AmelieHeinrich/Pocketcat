//
//  RTAO.swift
//  Pocketcat
//
//  Created by Amélie Heinrich on 24/03/2026.
//

import Metal
import simd

struct RTAOParameters {
    var frameIndex: UInt32 = 0
    var spp: UInt32 = 0
    var resolutionScale: Float = 0.0
    var aoRadius: Float = 1.0
}

class RTAO: Pass {
    private let pipeline: ComputePipeline
    private var visibilityMask: Texture
    private var accumulationFrame: UInt32 = 0
    private unowned var settings: SettingsRegistry

    init(settings: SettingsRegistry) {
        self.settings = settings
        self.settings.register(int: "RTAO.SamplesPerPixel", label: "Samples per pixel", default: 1, range: 1...32)
        self.settings.register(bool: "RTAO.Enabled", label: "Enabled", default: false)
        self.settings.register(enum: "RTAO.TracingResolution", label: "Tracing Resolution", default: TracingResolution.Quarter)
        self.settings.register(float: "RTAO.Radius", label: "AO Radius", default: 0.50, range: 0.1...10.0)

        pipeline = ComputePipeline(function: "rtao")

        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r8Unorm, width: 1, height: 1, mipmapped: false)
        desc.usage = [.shaderRead, .shaderWrite]
        let tex = Texture(descriptor: desc)
        tex.setLabel(name: "AO Mask")
        self.visibilityMask = tex

        super.init()
    }

    override func resize(renderWidth: Int, renderHeight: Int, outputWidth: Int, outputHeight: Int) {
        let resSetting = settings.enum("RTAO.TracingResolution", as: TracingResolution.self, default: .Quarter)
        let scale: Float
        
        switch resSetting {
        case .Quarter: scale = 0.25
        case .Half: scale = 0.5
        case .Native: scale = 1.0
        }

        visibilityMask.resize(width: Int(Float(renderWidth) * scale), height: Int(Float(renderHeight) * scale))
        accumulationFrame = 0
    }

    override func render(context: FrameContext) {
        guard context.scene != nil else { return }
        if !self.settings.bool("RTAO.Enabled") {
            return
        }

        let depth = context.resources.get("GBuffer.Depth") as Texture?
        let normal = context.resources.get("GBuffer.Normal") as Texture?

        guard let depth = depth, let normal = normal else { return }

        let resSetting = settings.enum("RTAO.TracingResolution", as: TracingResolution.self, default: .Quarter)
        let scale: Float
        
        switch resSetting {
        case .Quarter: scale = 0.25
        case .Half: scale = 0.5
        case .Native: scale = 1.0
        }

        let w = Int(Float(depth.texture.width) * scale)
        let h = Int(Float(depth.texture.height) * scale)

        if visibilityMask.texture.width != w || visibilityMask.texture.height != h {
            visibilityMask.resize(width: w, height: h)
            accumulationFrame = 0
        }

        let fi = accumulationFrame

        var parameters = RTAOParameters()
        parameters.frameIndex = fi
        parameters.spp = UInt32(settings.int("RTAO.SamplesPerPixel", default: 1))
        parameters.resolutionScale = scale
        parameters.aoRadius = settings.float("RTAO.Radius", default: 0.50)

        let cp = context.cmdBuffer.beginComputePass(name: "RTAO : Trace")
        cp.consumerBarrier(before: .dispatch, after: [.dispatch, .accelerationStructure])
        cp.setPipeline(pipeline: pipeline)
        cp.setBuffer(buf: context.sceneBuffer.buffer, index: 0)
        cp.setBytes(allocator: context.allocator, index: 1, bytes: &parameters, size: MemoryLayout<RTAOParameters>.size)
        cp.setTexture(texture: self.visibilityMask, index: 0)
        cp.setTexture(texture: depth, index: 1)
        cp.setTexture(texture: normal, index: 2)
        cp.dispatch(
            threads: MTLSizeMake((w + 7) / 8, (h + 7) / 8, 1), threadsPerGroup: MTLSizeMake(8, 8, 1)
        )
        cp.end()
        
        context.resources.register(self.visibilityMask, for: "RTAO.Mask")
        context.resources.addVisualizer(texture: visibilityMask, label: "RTAO",
            fragmentFunction: "texviz_single_channel_fs")

        accumulationFrame &+= 1
    }
}
