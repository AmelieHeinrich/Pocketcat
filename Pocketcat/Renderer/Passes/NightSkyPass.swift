//
//  NightSkyPass.swift
//  Pocketcat
//
//  Created by Amélie Heinrich on 15/05/2026.
//

import Metal
import simd

private struct NightStarParams {
    var starBrightness: Float
    var cubemapSize: UInt32
    var elapsedTime: Float
    var twinkleStrength: Float
}

private struct MilkyWayBakeParams {
    var milkywayExposure: Float
}

class NightSkyPass: Pass {
    static let cubemapSize = 1024
    static let milkywayCubemapSize = 1024

    private let starPipeline: RenderPipeline
    private let milkywayBakePipeline: ComputePipeline

    private let nightCubemap: Texture
    private let milkywayCubemap: Texture
    private let milkywayTex: Texture
    private let spriteTex: Texture

    private let starBuffer: Buffer
    private let starCount: Int

    private unowned var settings: SettingsRegistry

    init(settings: SettingsRegistry) {
        self.settings = settings
        settings.register(bool:  "NightSky.Enabled",          label: "Night Sky Enabled",  default: true)
        settings.register(float: "NightSky.MilkyWayExposure", label: "Milky Way Exposure", default: 0.03,  range: 0.001...1.0,   step: 0.001)
        settings.register(float: "NightSky.StarBrightness",   label: "Star Brightness",    default: 500.0, range: 1.0...5000.0,  step: 10.0)
        settings.register(float: "NightSky.TwinkleStrength",  label: "Twinkle Strength",   default: 0.35,  range: 0.0...2.0,     step: 0.05)

        milkywayBakePipeline = ComputePipeline(function: "nightsky_bake_milkyway", name: "NightSky Milky Way Bake")

        var starDesc = RenderPipelineDescriptor()
        starDesc.name             = "NightSky Stars"
        starDesc.vertexFunction   = "nightsky_stars_vs"
        starDesc.fragmentFunction = "nightsky_stars_fs"
        starDesc.pixelFormats     = [.rgba16Float]
        starDesc.additiveBlending = true
        starPipeline = RenderPipeline(descriptor: starDesc)

        // Stars-only cubemap rendered each night frame
        let cubeDesc = MTLTextureDescriptor()
        cubeDesc.textureType = .typeCube
        cubeDesc.pixelFormat = .rgba16Float
        cubeDesc.width       = NightSkyPass.cubemapSize
        cubeDesc.height      = NightSkyPass.cubemapSize
        cubeDesc.usage       = [.shaderRead, .shaderWrite, .renderTarget]
        nightCubemap = Texture(descriptor: cubeDesc)
        nightCubemap.setLabel(name: "Night Sky Cubemap")

        // Milky way cubemap baked once at startup from equirectangular source
        let mwCubeDesc = MTLTextureDescriptor()
        mwCubeDesc.textureType = .typeCube
        mwCubeDesc.pixelFormat = .rgba16Float
        mwCubeDesc.width       = NightSkyPass.milkywayCubemapSize
        mwCubeDesc.height      = NightSkyPass.milkywayCubemapSize
        mwCubeDesc.usage       = [.shaderRead, .shaderWrite]
        milkywayCubemap = Texture(descriptor: mwCubeDesc)
        milkywayCubemap.setLabel(name: "Milky Way Cubemap")

        // Load astronomy assets (baked .tex preferred, fall back to .png at runtime)
        let mw = TextureLoader.load(resource: "MilkyWayEquiRectangular", withExtension: "tex", label: "Milky Way")
            ?? TextureLoader.load(resource: "MilkyWayEquiRectangular", withExtension: "png", label: "Milky Way")
        milkywayTex = mw!
        milkywayTex.makeResident()

        let sprite = TextureLoader.load(resource: "star_airy_disk_sprite", withExtension: "tex", label: "Star Sprite")
            ?? TextureLoader.load(resource: "star_airy_disk_sprite", withExtension: "png", label: "Star Sprite")
        spriteTex = sprite!
        spriteTex.makeResident()

        let starURL = Bundle.main.url(forResource: "hyg_v42", withExtension: "star")!
        let result = HYGLoader(url: starURL).load()
        starBuffer = result.buffer
        starCount  = result.count

        RendererData.commitResidency()

        // One-shot milky way bake: equirectangular → cubemap with galactic transform
        let s = NightSkyPass.milkywayCubemapSize
        // Bake raw luminance (exposure = 1.0); composite_sky_cubemap applies the runtime exposure setting
        var bakeParams = MilkyWayBakeParams(milkywayExposure: 1.0)
        let cmdBuf = RendererData.mtl3commandQueue.makeCommandBuffer()!
        cmdBuf.label = "Milky Way Cubemap Bake"
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(milkywayBakePipeline.pipelineState)
        withUnsafeBytes(of: &bakeParams) { enc.setBytes($0.baseAddress!, length: $0.count, index: 0) }
        var cubeSize = UInt32(s)
        withUnsafeBytes(of: &cubeSize) { enc.setBytes($0.baseAddress!, length: $0.count, index: 1) }
        enc.setTexture(milkywayTex.texture, index: 0)
        enc.setTexture(milkywayCubemap.texture, index: 1)
        let tg = MTLSizeMake(8, 8, 1)
        enc.dispatchThreadgroups(MTLSizeMake((s + 7) / 8, (s + 7) / 8, 6), threadsPerThreadgroup: tg)
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        super.init()
    }

    override func render(context: FrameContext) {
        guard settings.bool("NightSky.Enabled", default: true) else {
            context.resources.register(nightCubemap, for: "Night.Cubemap")
            context.resources.register(milkywayCubemap, for: "Night.MilkyWayCubemap")
            return
        }

        if context.sunElevationDegrees > 10.0 {
            context.resources.register(nightCubemap, for: "Night.Cubemap")
            context.resources.register(milkywayCubemap, for: "Night.MilkyWayCubemap")
            return
        }

        let s = NightSkyPass.cubemapSize

        // Draw all stars into the cubemap in one layered render pass (vertex shader
        // selects the face via [[render_target_array_index]])
        var starParams = NightStarParams(
            starBrightness: settings.float("NightSky.StarBrightness", default: 500.0),
            cubemapSize: UInt32(s),
            elapsedTime: context.elapsedTime,
            twinkleStrength: settings.float("NightSky.TwinkleStrength", default: 0.35)
        )

        var rpDesc = RenderPassDescriptor()
        rpDesc.name                    = "NightSky Stars"
        rpDesc.renderTargetArrayLength = 6
        rpDesc.renderTargetWidth       = s
        rpDesc.renderTargetHeight      = s
        rpDesc.addAttachment(texture: nightCubemap,
                             clearColor: simd_float4(0, 0, 0, 1),
                             shouldClear: true)

        let rp = context.cmdBuffer.beginRenderPass(descriptor: rpDesc)
        rp.setPipeline(pipeline: starPipeline)
        rp.setBuffer(buf: starBuffer, index: 0, stages: [.vertex, .fragment])
        rp.setBytes(allocator: context.allocator, index: 1, bytes: &starParams, size: MemoryLayout<NightStarParams>.size, stages: [.vertex, .fragment])
        rp.setTexture(texture: spriteTex, index: 0, stages: .fragment)
        rp.setViewport(MTLViewport(originX: 0, originY: 0, width: Double(s), height: Double(s), znear: 0, zfar: 1))
        rp.drawInstanced(primitiveType: .triangle, vertexCount: 6, instanceCount: starCount)
        rp.end()

        context.resources.register(nightCubemap, for: "Night.Cubemap")
        context.resources.register(milkywayCubemap, for: "Night.MilkyWayCubemap")
        context.resources.addCubemapVisualizer(texture: nightCubemap, label: "Night Sky Cubemap")
    }
}
