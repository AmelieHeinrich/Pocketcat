//
//  MetalFXDesktopDenoiserPass.swift
//  Pocketcat
//
//  Temporal denoised upscaler for the desktop timeline.
//
//  Runs the MetalFX temporal denoised scaler on the noisy composited HDR signal,
//  denoising and upscaling to output resolution in a single step. Active only when
//  the "Upscaler.Type" setting is .TemporalDenoised. The RTGI texture is fed as the
//  diffuse albedo and the RTReflections texture as the specular albedo; the renderer
//  forces RTAO/RTGI/RTReflections on at Native tracing resolution while this mode is
//  selected (see EditorRendererController), so those resources always exist and match
//  the denoiser's input resolution.
//

import Metal
import MetalFX

class MetalFXDesktopDenoiserPass: Pass {
    private var denoiser: MTL4FXTemporalDenoisedScaler!
    private var outputTexture: Texture!
    private unowned let registry: SettingsRegistry
    private var firstFrame = true

    init(registry: SettingsRegistry) {
        self.registry = registry
        super.init()
    }

    override func resize(renderWidth: Int, renderHeight: Int, outputWidth: Int, outputHeight: Int) {
        let desc = MTLFXTemporalDenoisedScalerDescriptor()
        desc.colorTextureFormat  = .rgba16Float
        desc.outputTextureFormat = .rgba16Float
        desc.depthTextureFormat  = .depth32Float
        desc.motionTextureFormat = .rgba16Float
        desc.diffuseAlbedoTextureFormat  = .rgba8Unorm
        desc.normalTextureFormat = .rgba16Float
        desc.inputWidth   = renderWidth
        desc.inputHeight  = renderHeight
        desc.outputWidth  = outputWidth
        desc.outputHeight = outputHeight

        denoiser = desc.makeTemporalDenoisedScaler(device: RendererData.device, compiler: RendererData.compiler)!

        let texDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: outputWidth, height: outputHeight, mipmapped: false)
        texDesc.usage = [.shaderRead, .shaderWrite]
        texDesc.storageMode = .private
        outputTexture = Texture(descriptor: texDesc)
        outputTexture.setLabel(name: "MetalFX Desktop Denoised Output")

        firstFrame = true
    }

    override func render(context: FrameContext) {
        guard registry.enum("Upscaler.Type", as: UpscalerType.self, default: .None) == .TemporalDenoised
        else { return }

        guard let hdr      = context.resources.get("HDR")                   as Texture?,
              let depth    = context.resources.get("GBuffer.Depth")         as Texture?,
              let mv       = context.resources.get("GBuffer.MotionVectors") as Texture?,
              let normal   = context.resources.get("GBuffer.Normal")        as Texture?,
              let albedo   = context.resources.get("GBuffer.Albedo")        as Texture?
        else { return }

        denoiser.colorTexture          = hdr.texture
        denoiser.depthTexture          = depth.texture
        denoiser.motionTexture         = mv.texture
        denoiser.diffuseAlbedoTexture  = albedo.texture
        denoiser.normalTexture         = normal.texture
        denoiser.outputTexture         = outputTexture.texture
        denoiser.motionVectorScaleX    = Float(hdr.texture.width)
        denoiser.motionVectorScaleY    = Float(hdr.texture.height)

        context.cmdBuffer.pushMarker(name: "MetalFX Temporal Denoise")
        denoiser.encode(commandBuffer: context.cmdBuffer.commandBuffer)
        context.cmdBuffer.popMarker()

        firstFrame = false
        context.resources.register(outputTexture, for: "HDR")
        context.resources.register(outputTexture, for: "Denoiser.Upscaled")
    }
}
