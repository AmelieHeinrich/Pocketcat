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

struct RTShadowTemporalDenoiseInput {
    var shadow_mask: MTLResourceID
    var motion_vectors: MTLResourceID
    var current_normals: MTLResourceID
    var previous_normals: MTLResourceID
    var filtered: MTLResourceID
    var previous_filtered: MTLResourceID
    var moments: MTLResourceID
    var previous_moments: MTLResourceID
    var history: MTLResourceID
    var depth_threshold: Float
    var normal_threshold: Float
}

struct RTShadowVarianceEstimationInput {
    var input_shadow: MTLResourceID
    var normals: MTLResourceID
    var motion_vectors: MTLResourceID
    var moments: MTLResourceID
    var history: MTLResourceID
    var output_shadow: MTLResourceID
    var phi_color: Float
    var phi_normal: Float
}

struct RTShadowAtrousInput {
    var input_shadow: MTLResourceID
    var motion_vectors: MTLResourceID
    var normals: MTLResourceID
    var output_shadow: MTLResourceID
    var step_size: Int32
    var phi_color: Float
    var phi_normal: Float
}

class RTShadows: Pass {
    private let tracePipeline: ComputePipeline
    private let temporalPipeline: ComputePipeline
    private let variancePipeline: ComputePipeline
    private let atrousPipeline: ComputePipeline
    private var ift: MTLIntersectionFunctionTable

    // Raw trace output
    private var shadowMask: Texture

    // Temporal / à-trous ping-pong outputs
    private var filtered: [Texture]
    private var moments: [Texture]

    // History (updated each frame before à-trous runs)
    private var prevFiltered: Texture
    private var prevMoments: Texture
    private var historyLength: Texture

    private var ping: Int = 0
    private var pong: Int = 1
    private var accumulationFrame: UInt32 = 0
    private unowned var settings: SettingsRegistry

    let atrousIterations = 3

    init(settings: SettingsRegistry) {
        self.settings = settings
        self.settings.register(
            int: "RTShadows.SamplesPerPixel", label: "Samples per pixel", default: 1, range: 1...32)
        self.settings.register(
            bool: "RTShadows.DenoisingEnabled", label: "Enable Denoising", default: false)
        self.settings.register(
            float: "RTShadows.Denoiser.DepthThreshold", label: "Temporal Depth Threshold",
            default: 0.15, range: 0.01...1.0, step: 0.01)
        self.settings.register(
            float: "RTShadows.Denoiser.NormalThreshold", label: "Temporal Normal Threshold",
            default: 0.36, range: 0.0...1.0, step: 0.01)
        self.settings.register(
            float: "RTShadows.Denoiser.ColorPhi", label: "Color Phi", default: 10.0,
            range: 1.0...100.0, step: 1.0)
        self.settings.register(
            float: "RTShadows.Denoiser.NormalPhi", label: "Normal Phi", default: 128.0,
            range: 1.0...256.0, step: 1.0)

        tracePipeline = ComputePipeline(function: "rt_shadows", linkedFunctions: ["alpha_any_hit"])
        temporalPipeline = ComputePipeline(function: "denoise_shadows_temporal")
        variancePipeline = ComputePipeline(function: "denoise_shadows_variance_estimation")
        atrousPipeline = ComputePipeline(function: "denoise_shadows_atrous")
        ift = tracePipeline.createIFT()

        func makeTex(_ fmt: MTLPixelFormat, _ label: String) -> Texture {
            let desc = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: fmt, width: 1, height: 1, mipmapped: false)
            desc.usage = [.shaderRead, .shaderWrite]
            let t = Texture(descriptor: desc)
            t.setLabel(name: label)
            return t
        }

        shadowMask = makeTex(.r8Unorm, "RTShadows.NoisyMask")
        filtered = [
            makeTex(.rg16Float, "RTShadows.Filtered.Ping"),
            makeTex(.rg16Float, "RTShadows.Filtered.Pong"),
        ]
        moments = [
            makeTex(.rgba16Float, "RTShadows.Moments.Ping"),
            makeTex(.rgba16Float, "RTShadows.Moments.Pong"),
        ]
        prevFiltered = makeTex(.rg16Float, "RTShadows.Prev.Filtered")
        prevMoments = makeTex(.rgba16Float, "RTShadows.Prev.Moments")
        historyLength = makeTex(.r16Float, "RTShadows.HistoryLength")

        super.init()
    }

    override func resize(renderWidth: Int, renderHeight: Int, outputWidth: Int, outputHeight: Int) {
        shadowMask.resize(width: renderWidth, height: renderHeight)
        for t in filtered { t.resize(width: renderWidth, height: renderHeight) }
        for t in moments { t.resize(width: renderWidth, height: renderHeight) }
        prevFiltered.resize(width: renderWidth, height: renderHeight)
        prevMoments.resize(width: renderWidth, height: renderHeight)
        historyLength.resize(width: renderWidth, height: renderHeight)
        accumulationFrame = 0
    }

    override func render(context: FrameContext) {
        guard context.scene != nil else { return }

        traceShadowRays(context: context)
        denoise(context: context)

        accumulationFrame &+= 1
    }

    private func traceShadowRays(context: FrameContext) {
        let w = shadowMask.texture.width
        let h = shadowMask.texture.height

        let depth = context.resources.get("GBuffer.Depth") as Texture?
        let normals = context.resources.get("GBuffer.Normal") as Texture?
        guard let depth = depth, let normals = normals else { return }

        var parameters = RTShadowParameters()
        parameters.spp = UInt32(settings.int("RTShadows.SamplesPerPixel", default: 1))
        parameters.frameIndex = accumulationFrame

        ift.setBuffer(context.sceneBuffer.buffer.buffer, offset: 0, index: 0)

        let cp = context.cmdBuffer.beginComputePass(name: "RT Shadows: Trace")
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
        cp.end()

        if !settings.bool("RTShadows.DenoisingEnabled") {
            context.resources.register(shadowMask, for: "RTShadows.Output")
        }
    }

    private func denoise(context: FrameContext) {
        if !settings.bool("RTShadows.DenoisingEnabled") {
            return
        }

        let w = filtered[ping].texture.width
        let h = filtered[ping].texture.height

        let motionVectors = context.resources.get("GBuffer.MotionVectors") as Texture?
        let normals = context.resources.get("GBuffer.Normal") as Texture?
        let previousNormals = context.resources.get("History.GBuffer.Normal") as Texture?
        guard let motionVectors = motionVectors, let normals = normals,
            let previousNormals = previousNormals
        else { return }

        // 1. Temporal accumulation → filtered[ping], moments[ping]
        var temporalInput = RTShadowTemporalDenoiseInput(
            shadow_mask: shadowMask.texture.gpuResourceID,
            motion_vectors: motionVectors.texture.gpuResourceID,
            current_normals: normals.texture.gpuResourceID,
            previous_normals: previousNormals.texture.gpuResourceID,
            filtered: filtered[ping].texture.gpuResourceID,
            previous_filtered: prevFiltered.texture.gpuResourceID,
            moments: moments[ping].texture.gpuResourceID,
            previous_moments: prevMoments.texture.gpuResourceID,
            history: historyLength.texture.gpuResourceID,
            depth_threshold: settings.float("RTShadows.Denoiser.DepthThreshold", default: 0.15),
            normal_threshold: settings.float("RTShadows.Denoiser.NormalThreshold", default: 0.36)
        )

        // Temporal
        let cp = context.cmdBuffer.beginComputePass(name: "RT Shadows : Denoise")
        cp.consumerBarrier(before: .dispatch, after: .dispatch)
        cp.setPipeline(pipeline: temporalPipeline)
        cp.setBytes(allocator: context.allocator, index: 0, bytes: &temporalInput, size: MemoryLayout<RTShadowTemporalDenoiseInput>.size)
        cp.dispatch(
            threads: MTLSizeMake((w + 7) / 8, (h + 7) / 8, 1), threadsPerGroup: MTLSizeMake(8, 8, 1)
        )

        // Copy history
        cp.intraPassBarrier(before: .blit, after: .dispatch)
        cp.copyTexture(src: filtered[ping], dst: prevFiltered)
        cp.copyTexture(src: moments[ping], dst: prevMoments)

        // Variance Estimation
        var varianceInput = RTShadowVarianceEstimationInput(
            input_shadow: filtered[ping].texture.gpuResourceID,
            normals: normals.texture.gpuResourceID,
            motion_vectors: motionVectors.texture.gpuResourceID,
            moments: moments[ping].texture.gpuResourceID,
            history: historyLength.texture.gpuResourceID,
            output_shadow: filtered[pong].texture.gpuResourceID,
            phi_color: settings.float("RTShadows.Denoiser.ColorPhi", default: 10.0),
            phi_normal: settings.float("RTShadows.Denoiser.NormalPhi", default: 128.0)
        )
        cp.intraPassBarrier(before: .dispatch, after: [.dispatch, .blit])
        cp.setPipeline(pipeline: variancePipeline)
        cp.setBytes(allocator: context.allocator, index: 0, bytes: &varianceInput, size: MemoryLayout<RTShadowVarianceEstimationInput>.size)
        cp.dispatch(
            threads: MTLSizeMake((w + 7) / 8, (h + 7) / 8, 1), threadsPerGroup: MTLSizeMake(8, 8, 1)
        )

        // Spatial
        var readIdx = pong
        var writeIdx = ping
        for i in 0..<atrousIterations {
            var atrousInput = RTShadowAtrousInput(
                input_shadow: filtered[readIdx].texture.gpuResourceID,
                motion_vectors: motionVectors.texture.gpuResourceID,
                normals: normals.texture.gpuResourceID,
                output_shadow: filtered[writeIdx].texture.gpuResourceID,
                step_size: Int32(1 << i),
                phi_color: settings.float("RTShadows.Denoiser.ColorPhi", default: 10.0),
                phi_normal: settings.float("RTShadows.Denoiser.NormalPhi", default: 128.0)
            )
            cp.intraPassBarrier(before: .dispatch, after: [.dispatch, .blit])
            cp.setPipeline(pipeline: atrousPipeline)
            cp.setBytes(
                allocator: context.allocator, index: 0, bytes: &atrousInput,
                size: MemoryLayout<RTShadowAtrousInput>.size)
            cp.dispatch(
                threads: MTLSizeMake((w + 7) / 8, (h + 7) / 8, 1),
                threadsPerGroup: MTLSizeMake(8, 8, 1)
            )
            swap(&readIdx, &writeIdx)
        }
        cp.end()

        swap(&ping, &pong)

        context.resources.register(filtered[readIdx], for: "RTShadows.Output")
    }
}
