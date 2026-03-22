import Metal
import MetalFX
internal import QuartzCore

enum UpscalerType: Int, CaseIterable {
    case None = 0
    case Spatial = 1
    case Temporal = 2
}

class MetalFXUpscalePass: Pass {
    var spatialUpscaler: MTL4FXSpatialScaler!
    var temporalUpscaler: MTL4FXTemporalScaler!
    unowned let registry: SettingsRegistry
    var firstFrameTemporal = true

    init(registry: SettingsRegistry) {
        self.registry = registry
        registry.register(enum: "Upscaler.Type", label: "Upscaler", default: UpscalerType.Temporal)
        super.init()
    }

    override func resize(renderWidth: Int, renderHeight: Int, outputWidth: Int, outputHeight: Int) {
        let spatialDesc = MTLFXSpatialScalerDescriptor()
        spatialDesc.colorProcessingMode = .perceptual
        spatialDesc.colorTextureFormat = .bgra8Unorm
        spatialDesc.outputTextureFormat = .bgra8Unorm
        spatialDesc.inputWidth = renderWidth
        spatialDesc.inputHeight = renderHeight
        spatialDesc.outputWidth = outputWidth
        spatialDesc.outputHeight = outputHeight

        self.spatialUpscaler = spatialDesc.makeSpatialScaler(device: RendererData.device, compiler: RendererData.compiler)!
        
        let temporalDesc = MTLFXTemporalScalerDescriptor()
        temporalDesc.colorTextureFormat = .bgra8Unorm
        temporalDesc.outputTextureFormat = .bgra8Unorm
        temporalDesc.depthTextureFormat = .depth32Float
        temporalDesc.motionTextureFormat = .rg16Float
        temporalDesc.inputWidth = renderWidth
        temporalDesc.inputHeight = renderHeight
        temporalDesc.outputWidth = outputWidth
        temporalDesc.outputHeight = outputHeight
        temporalDesc.requiresSynchronousInitialization = true
        
        self.temporalUpscaler = temporalDesc.makeTemporalScaler(device: RendererData.device, compiler: RendererData.compiler)!
        firstFrameTemporal = true
    }

    override func render(context: FrameContext) {
        let type = registry.enum("Upscaler.Type", as: UpscalerType.self, default: .Temporal)
        switch type {
        case .None: break
        case .Spatial:
            let ldrResource = context.resources.get("LDR") as Texture?
            guard let ldr = ldrResource else { return }
            
            spatialUpscaler.colorTexture = ldr.texture
            spatialUpscaler.outputTexture = context.drawable.texture
            
            context.cmdBuffer.pushMarker(name: "MetalFX Spatial")
            spatialUpscaler.encode(commandBuffer: context.cmdBuffer.commandBuffer)
            context.cmdBuffer.popMarker()
        case .Temporal:
            let ldrResource = context.resources.get("LDR") as Texture?
            let depthResource = context.resources.get("GBuffer.Depth") as Texture?
            let mvResource = context.resources.get("GBuffer.MotionVectors") as Texture?
            
            guard let ldr = ldrResource else { return }
            guard let depth = depthResource else { return }
            guard let mv = mvResource else { return }
            
            temporalUpscaler.colorTexture = ldr.texture
            temporalUpscaler.outputTexture = context.drawable.texture
            temporalUpscaler.depthTexture = depth.texture
            temporalUpscaler.motionTexture = mv.texture
            temporalUpscaler.reset = firstFrameTemporal
            temporalUpscaler.motionVectorScaleX = Float(ldr.texture.width)
            temporalUpscaler.motionVectorScaleY = Float(ldr.texture.height)
            
            context.cmdBuffer.pushMarker(name: "MetalFX Temporal")
            temporalUpscaler.encode(commandBuffer: context.cmdBuffer.commandBuffer)
            context.cmdBuffer.popMarker()
            
            firstFrameTemporal = false
            break
        }
    }
}
