//
//  RTGI.swift
//  Pocketcat
//
//  Created by Amélie Heinrich on 23/03/2026.
//

import Metal
import simd

enum TracingResolution: Int, CaseIterable {
    case Quarter = 0
    case Half = 1
    case Native = 2
}

struct RTGIParameters {
    var frameIndex: UInt32 = 0
    var spp: UInt32 = 0
    var resolutionScale: Float = 0.0
    var padding: UInt32 = 0
}

class RTGI: Pass {
    private let pipeline: ComputePipeline
    private let diffuseTexture: Texture
    private var accumulationFrame: UInt32 = 0
    private unowned var settings: SettingsRegistry

    init(settings: SettingsRegistry) {
        self.settings = settings
        self.settings.register(int: "RTGI.SamplesPerPixel", label: "Samples per pixel", default: 1, range: 1...32)
        self.settings.register(bool: "RTGI.Enabled", label: "Enabled", default: false)
        self.settings.register(enum: "RTGI.TracingResolution", label: "Tracing Resolution", default: TracingResolution.Quarter)

        pipeline = ComputePipeline(function: "rtgi")

        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 1, height: 1, mipmapped: false)
        desc.usage = [.shaderRead, .shaderWrite]
        let tex = Texture(descriptor: desc)
        tex.setLabel(name: "Diffuse Indirect")
        self.diffuseTexture = tex

        super.init()
    }

    override func resize(renderWidth: Int, renderHeight: Int, outputWidth: Int, outputHeight: Int) {
        let resSetting = settings.enum(
            "RTGI.TracingResolution", as: TracingResolution.self, default: .Quarter)
        let scale: Float
        switch resSetting {
        case .Quarter: scale = 0.25
        case .Half: scale = 0.5
        case .Native: scale = 1.0
        }

        diffuseTexture.resize(
            width: Int(Float(renderWidth) * scale), height: Int(Float(renderHeight) * scale))
        accumulationFrame = 0
    }

    override func render(context: FrameContext) {
        guard context.scene != nil else { return }
        if !self.settings.bool("RTGI.Enabled") {
            return
        }

        let depth = context.resources.get("GBuffer.Depth") as Texture?
        let albedo = context.resources.get("GBuffer.Albedo") as Texture?
        let normal = context.resources.get("GBuffer.Normal") as Texture?

        guard let depth = depth, let albedo = albedo, let normal = normal else { return }

        let resSetting = settings.enum("RTGI.TracingResolution", as: TracingResolution.self, default: .Quarter)
        let scale: Float
        switch resSetting {
        case .Quarter: scale = 0.25
        case .Half: scale = 0.5
        case .Native: scale = 1.0
        }

        let w = Int(Float(depth.texture.width) * scale)
        let h = Int(Float(depth.texture.height) * scale)

        if diffuseTexture.texture.width != w || diffuseTexture.texture.height != h {
            diffuseTexture.resize(width: w, height: h)
            accumulationFrame = 0
        }

        let fi = accumulationFrame

        var parameters = RTGIParameters()
        parameters.frameIndex = fi
        parameters.spp = UInt32(settings.int("RTGI.SamplesPerPixel", default: 1))
        parameters.resolutionScale = scale

        let cp = context.cmdBuffer.beginComputePass(name: "RTGI : Trace")
        cp.consumerBarrier(before: .dispatch, after: [.dispatch, .accelerationStructure])
        cp.setPipeline(pipeline: pipeline)
        cp.setBuffer(buf: context.sceneBuffer.buffer, index: 0)
        cp.setBytes(allocator: context.allocator, index: 1, bytes: &parameters, size: MemoryLayout<RTGIParameters>.size)
        cp.setTexture(texture: self.diffuseTexture, index: 0)
        cp.setTexture(texture: depth, index: 1)
        cp.setTexture(texture: normal, index: 2)
        cp.setTexture(texture: albedo, index: 3)
        cp.dispatch(
            threads: MTLSizeMake((w + 7) / 8, (h + 7) / 8, 1), threadsPerGroup: MTLSizeMake(8, 8, 1)
        )
        cp.end()
        
        context.resources.register(self.diffuseTexture, for: "RTGI.Texture")
        context.resources.addVisualizer(texture: diffuseTexture, label: "RTGI",
            fragmentFunction: "texviz_hdr_fs")

        accumulationFrame &+= 1
    }
}
