import Metal
import MetalFX

class MetalFXDenoiserPass: Pass {
    private var denoiser: MTL4FXTemporalDenoisedScaler!
    private var outputTexture: Texture!
    private var firstFrame = true

    override init() { super.init() }

    override func resize(renderWidth: Int, renderHeight: Int, outputWidth: Int, outputHeight: Int) {
        let desc = MTLFXTemporalDenoisedScalerDescriptor()
        desc.colorTextureFormat  = .rgba16Float
        desc.outputTextureFormat = .rgba16Float
        desc.depthTextureFormat  = .depth32Float
        desc.motionTextureFormat = .rgba16Float
        desc.diffuseAlbedoTextureFormat = .rgba8Unorm
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
        outputTexture.setLabel(name: "MetalFX Denoised Output")

        firstFrame = true
    }

    override func render(context: FrameContext) {
        guard let raw    = context.resources.get("PT.RawSample")          as Texture?,
              let depth  = context.resources.get("GBuffer.Depth")         as Texture?,
              let mv     = context.resources.get("GBuffer.MotionVectors")  as Texture?,
              let albedo = context.resources.get("GBuffer.Albedo")        as Texture?,
              let normal = context.resources.get("GBuffer.Normal")        as Texture?
        else { return }

        denoiser.colorTexture          = raw.texture
        denoiser.depthTexture          = depth.texture
        denoiser.motionTexture         = mv.texture
        denoiser.diffuseAlbedoTexture  = albedo.texture
        denoiser.specularAlbedoTexture = albedo.texture
        denoiser.normalTexture         = normal.texture
        denoiser.outputTexture         = outputTexture.texture
        denoiser.motionVectorScaleX    = Float(raw.texture.width)
        denoiser.motionVectorScaleY    = Float(raw.texture.height)

        context.cmdBuffer.pushMarker(name: "MetalFX Denoise")
        denoiser.encode(commandBuffer: context.cmdBuffer.commandBuffer)
        context.cmdBuffer.popMarker()

        firstFrame = false
        context.resources.register(outputTexture, for: "HDR")
        context.resources.register(outputTexture, for: "Denoiser.Upscaled")
    }
}
