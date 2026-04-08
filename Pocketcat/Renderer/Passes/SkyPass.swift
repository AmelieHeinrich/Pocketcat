//
//  SkyPass.swift
//  Pocketcat
//
//  Created by Amélie Heinrich on 03/04/2026.
//

import Metal
import simd

// Mirror of atmosphere_parameters in Sky.metal — must stay in sync.
// Uses float4/simd_float4 for float3 fields to guarantee correct alignment (Swift simd_float3 has
// alignment 4 while Metal float3 has alignment 16; float4 has alignment 16 in both).
struct AtmosphereParameters {
    var rayleighScattering: simd_float4 = simd_float4(5.802, 13.558, 33.1, 0)  // Mm^-1
    var rayleighScaleHeight: Float = 8.0  // km
    var rayleighAbsorptionBase: Float = 0.0
    var mieScattering: Float = 3.996  // Mm^-1
    var mieAbsorption: Float = 4.4  // Mm^-1
    var mieScaleHeight: Float = 1.2  // km
    var miePhaseG: Float = 0.8
    var _pad0: simd_float2 = .zero  // explicit padding to match Metal layout
    var ozoneAbsorption: simd_float4 = simd_float4(0.650, 1.881, 0.085, 0)  // Mm^-1
    var groundAlbedo: simd_float4 = simd_float4(0.3, 0.3, 0.3, 0)
    var groundRadiusMm: Float = 6.36
    var atmosphereRadiusMm: Float = 6.46
    var skyIntensity: Float = 5.0
    var sunDiskIntensity: Float = 20.0
    var sunDiskSize: Float = 2.0  // degrees
}

class SkyPass: Pass {
    private let transmittancePipeline: ComputePipeline
    private let multipleScatteringPipeline: ComputePipeline
    private let skyViewPipeline: ComputePipeline
    private let cubemapPipeline: ComputePipeline

    // Fixed-size LUT textures (not viewport-dependent)
    private let transmittanceLUT: Texture  // 256×64
    private let multipleScatteringLUT: Texture  // 32×32
    private let skyViewLUT: Texture  // 200×100
    private let skyCubemap: Texture  // cube 128×128
    private var lutBaked: Bool = false

    private unowned var settings: SettingsRegistry

    init(settings: SettingsRegistry) {
        self.settings = settings
        settings.register(bool: "Sky.Enabled", label: "Sky Enabled", default: true)
        settings.register(
            float: "Sky.MiePhaseG", label: "Sky Mie Phase G", default: 0.8, range: 0.0...0.999,
            step: 0.01)
        settings.register(
            float: "Sky.Intensity", label: "Sky Intensity", default: 5.0, range: 0.1...50.0,
            step: 0.1)
        settings.register(
            float: "Sky.SunDiskIntensity", label: "Sun Disk Intensity", default: 20.0,
            range: 0.0...500.0, step: 1.0)
        settings.register(
            float: "Sky.SunDiskSize", label: "Sun Disk Size (deg)", default: 2.0, range: 0.5...20.0,
            step: 0.1)

        transmittancePipeline = ComputePipeline(function: "transmittance_lut", name: "Sky Transmittance LUT")
        multipleScatteringPipeline = ComputePipeline(function: "multiple_scattering_lut", name: "Sky Multiple Scattering LUT")
        skyViewPipeline = ComputePipeline(function: "sky_view_lut", name: "Sky View LUT")
        cubemapPipeline = ComputePipeline(function: "bake_skybox_cubemap", name: "Sky Cubemap Bake")

        let tlutDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 256, height: 64, mipmapped: false)
        tlutDesc.usage = [.shaderRead, .shaderWrite]
        transmittanceLUT = Texture(descriptor: tlutDesc)
        transmittanceLUT.setLabel(name: "Sky Transmittance LUT")

        let msDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 32, height: 32, mipmapped: false)
        msDesc.usage = [.shaderRead, .shaderWrite]
        multipleScatteringLUT = Texture(descriptor: msDesc)
        multipleScatteringLUT.setLabel(name: "Sky Multiple Scattering LUT")

        let svDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 200, height: 100, mipmapped: false)
        svDesc.usage = [.shaderRead, .shaderWrite]
        skyViewLUT = Texture(descriptor: svDesc)
        skyViewLUT.setLabel(name: "Sky View LUT")

        let cubeDesc = MTLTextureDescriptor()
        cubeDesc.textureType = .typeCube
        cubeDesc.pixelFormat = .rgba16Float
        cubeDesc.width = 128
        cubeDesc.height = 128
        cubeDesc.usage = [.shaderRead, .shaderWrite]
        skyCubemap = Texture(descriptor: cubeDesc)
        skyCubemap.setLabel(name: "Sky Cubemap")

        super.init()
    }

    override func render(context: FrameContext) {
        guard context.scene != nil else { return }
        guard settings.bool("Sky.Enabled", default: true) else { return }

        var params = AtmosphereParameters()
        params.miePhaseG = settings.float("Sky.MiePhaseG", default: 0.8)
        params.skyIntensity = settings.float("Sky.Intensity", default: 5.0)
        params.sunDiskIntensity = settings.float("Sky.SunDiskIntensity", default: 20.0)
        params.sunDiskSize = settings.float("Sky.SunDiskSize", default: 2.0)

        let cp = context.cmdBuffer.beginComputePass(name: "Sky")

        // 1. Transmittance LUT — buffer(0) = atmosphere_parameters, texture(0) = output
        if !lutBaked {
            cp.pushMarker(name: "Transmittance LUT")
            cp.setPipeline(pipeline: transmittancePipeline)
            cp.setBytes(allocator: context.allocator, index: 0, bytes: &params, size: MemoryLayout<AtmosphereParameters>.size)
            cp.setTexture(texture: transmittanceLUT, index: 0)
            cp.dispatch(threads: MTLSizeMake((256 + 15) / 16, (64 + 7) / 8, 1), threadsPerGroup: MTLSizeMake(16, 8, 1))
            cp.popMarker()

            // 2. Multiple Scattering LUT — texture(0) = tlut, texture(1) = output, buffer(0) = params
            cp.intraPassBarrier(before: .dispatch, after: .dispatch)
            cp.pushMarker(name: "Multiple Scattering LUT")
            cp.setPipeline(pipeline: multipleScatteringPipeline)
            cp.setBytes(allocator: context.allocator, index: 0, bytes: &params, size: MemoryLayout<AtmosphereParameters>.size)
            cp.setBuffer(buf: context.sceneBuffer.buffer, index: 1)
            cp.setTexture(texture: transmittanceLUT, index: 0)
            cp.setTexture(texture: multipleScatteringLUT, index: 1)
            cp.dispatch(threads: MTLSizeMake((32 + 7) / 8, (32 + 7) / 8, 1), threadsPerGroup: MTLSizeMake(8, 8, 1))
            cp.popMarker()
            cp.intraPassBarrier(before: .dispatch, after: .dispatch)
            
            lutBaked = true
        }

        // 3. Sky View LUT — texture(0) = tlut, texture(1) = mslut, texture(2) = output
        //                   buffer(0) = scene_data, buffer(1) = atmosphere_parameters
        cp.pushMarker(name: "Sky View LUT")
        cp.setPipeline(pipeline: skyViewPipeline)
        cp.setBuffer(buf: context.sceneBuffer.buffer, index: 0)
        cp.setBytes(allocator: context.allocator, index: 1, bytes: &params, size: MemoryLayout<AtmosphereParameters>.size)
        cp.setTexture(texture: transmittanceLUT, index: 0)
        cp.setTexture(texture: multipleScatteringLUT, index: 1)
        cp.setTexture(texture: skyViewLUT, index: 2)
        cp.dispatch(threads: MTLSizeMake((200 + 7) / 8, (100 + 7) / 8, 1), threadsPerGroup: MTLSizeMake(8, 8, 1))
        cp.popMarker()

        cp.intraPassBarrier(before: .dispatch, after: .dispatch)
        cp.pushMarker(name: "Cubemap Bake")
        cp.setPipeline(pipeline: cubemapPipeline)
        cp.setBuffer(buf: context.sceneBuffer.buffer, index: 0)
        cp.setBytes(allocator: context.allocator, index: 1, bytes: &params, size: MemoryLayout<AtmosphereParameters>.size)
        cp.setTexture(texture: skyViewLUT, index: 0)
        cp.setTexture(texture: skyCubemap, index: 1)
        cp.setTexture(texture: transmittanceLUT, index: 2)
        cp.dispatch(threads: MTLSizeMake((128 + 7) / 8, (128 + 7) / 8, 6), threadsPerGroup: MTLSizeMake(8, 8, 1))
        cp.popMarker()

        cp.end()

        context.resources.register(skyViewLUT, for: "Sky.ViewLUT")
        context.resources.register(skyCubemap, for: "Sky.Cubemap")
        context.resources.register(transmittanceLUT, for: "Sky.TransmittanceLUT")
        context.sceneBuffer.setSkybox(skyCubemap)
        
        context.resources.addCubemapVisualizer(texture: skyCubemap, label: "Sky Cubemap")
    }
}
