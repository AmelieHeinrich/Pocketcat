//
//  RTReflections.swift
//  Pocketcat
//
//  Created by Amélie Heinrich on 27/03/2026.
//

import Metal
import simd

struct RTReflectionsParameters {
    var frameIndex: UInt32 = 0
    var spp: UInt32 = 0
    var resolutionScale: Float = 0.0
    var metallicThreshold: Float = 0.9
}

class RTReflections: Pass {
    private let pipeline: ComputePipeline
    private let specularTexture: Texture
    private var accumulationFrame: UInt32 = 0
    private unowned var settings: SettingsRegistry

    init(settings: SettingsRegistry) {
        self.settings = settings
        self.settings.register(bool: "RTReflections.Enabled", label: "Enabled", default: false)
        self.settings.register(int: "RTReflections.SamplesPerPixel", label: "Samples per pixel", default: 1, range: 1...32)
        self.settings.register(enum: "RTReflections.TracingResolution", label: "Tracing Resolution", default: TracingResolution.Half)
        self.settings.register(float: "RTReflections.MetallicThreshold", label: "Metallic Threshold", default: 0.9, range: 0.0...1.0)

        pipeline = ComputePipeline(function: "rt_reflections")

        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 1, height: 1, mipmapped: false)
        desc.usage = [.shaderRead, .shaderWrite]
        let tex = Texture(descriptor: desc)
        tex.setLabel(name: "Specular Indirect")
        self.specularTexture = tex

        super.init()
    }

    override func resize(renderWidth: Int, renderHeight: Int, outputWidth: Int, outputHeight: Int) {
        let resSetting = settings.enum(
            "RTReflections.TracingResolution", as: TracingResolution.self, default: .Half)
        let scale: Float
        switch resSetting {
        case .Quarter: scale = 0.25
        case .Half:    scale = 0.5
        case .Native:  scale = 1.0
        }

        specularTexture.resize(
            width: Int(Float(renderWidth) * scale), height: Int(Float(renderHeight) * scale))
        accumulationFrame = 0
    }

    override func render(context: FrameContext) {
        guard context.scene != nil else { return }
        if !self.settings.bool("RTReflections.Enabled") {
            return
        }

        let depth  = context.resources.get("GBuffer.Depth")  as Texture?
        let normal = context.resources.get("GBuffer.Normal") as Texture?
        let albedo = context.resources.get("GBuffer.Albedo") as Texture?
        let orm    = context.resources.get("GBuffer.ORM")    as Texture?

        guard let depth = depth, let normal = normal, let albedo = albedo, let orm = orm else { return }

        let resSetting = settings.enum("RTReflections.TracingResolution", as: TracingResolution.self, default: .Half)
        let scale: Float
        switch resSetting {
        case .Quarter: scale = 0.25
        case .Half:    scale = 0.5
        case .Native:  scale = 1.0
        }

        let w = Int(Float(depth.texture.width) * scale)
        let h = Int(Float(depth.texture.height) * scale)

        if specularTexture.texture.width != w || specularTexture.texture.height != h {
            specularTexture.resize(width: w, height: h)
            accumulationFrame = 0
        }

        var parameters = RTReflectionsParameters()
        parameters.frameIndex        = accumulationFrame
        parameters.spp               = UInt32(settings.int("RTReflections.SamplesPerPixel", default: 1))
        parameters.resolutionScale   = scale
        parameters.metallicThreshold = settings.float("RTReflections.MetallicThreshold", default: 0.9)

        let cp = context.cmdBuffer.beginComputePass(name: "RT Reflections : Trace")
        cp.consumerBarrier(before: .dispatch, after: [.dispatch, .accelerationStructure])
        cp.setPipeline(pipeline: pipeline)
        cp.setBuffer(buf: context.sceneBuffer.buffer, index: 0)
        cp.setBytes(allocator: context.allocator, index: 1, bytes: &parameters, size: MemoryLayout<RTReflectionsParameters>.size)
        cp.setTexture(texture: self.specularTexture, index: 0)
        cp.setTexture(texture: depth,  index: 1)
        cp.setTexture(texture: normal, index: 2)
        cp.setTexture(texture: albedo, index: 3)
        cp.setTexture(texture: orm,    index: 4)
        cp.dispatch(
            threads: MTLSizeMake((w + 7) / 8, (h + 7) / 8, 1), threadsPerGroup: MTLSizeMake(8, 8, 1)
        )
        cp.end()

        context.resources.register(self.specularTexture, for: "RTReflections.Texture")
        context.resources.addVisualizer(texture: specularTexture, label: "RTReflections",
            fragmentFunction: "texviz_hdr_fs")

        accumulationFrame &+= 1
    }
}
