import Metal
internal import QuartzCore
import simd

enum TonemapMode: Int, CaseIterable {
    case AGX       = 0
    case ACES      = 1
    case Reinhard  = 2
    case Uncharted2 = 3
    case None      = 4
}

private struct TonemapParams {
    var gamma: Float
    var mode: Int32
}

class TonemapPass: Pass {
    private let pipeline: RenderPipeline
    private unowned let registry: SettingsRegistry
    private var ldrTexture: Texture?

    init(registry: SettingsRegistry) {
        var pipelineDesc = RenderPipelineDescriptor()
        pipelineDesc.name = "Tonemap"
        pipelineDesc.vertexFunction = "tonemap_vs"
        pipelineDesc.fragmentFunction = "tonemap_fs"
        pipelineDesc.pixelFormats = [.bgra8Unorm]

        self.pipeline = RenderPipeline(descriptor: pipelineDesc)
        self.registry = registry
        registry.register(enum: "Tonemap.Mode", label: "Algorithm", default: TonemapMode.ACES)
        registry.register(float: "Tonemap.Gamma", label: "Gamma", default: 1.4, range: 1.0...3.0)

        super.init()
    }

    override func resize(renderWidth: Int, renderHeight: Int, outputWidth: Int, outputHeight: Int) {
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: renderWidth, height: renderHeight, mipmapped: false)
        desc.usage = [.renderTarget, .shaderRead]
        ldrTexture = Texture(descriptor: desc)
    }

    override func render(context: FrameContext) {
        let forward = context.resources.get("HDR") as Texture?
        guard let forward = forward else { return }

        let mode = registry.enum("Tonemap.Mode", as: TonemapMode.self, default: .AGX)
        var params = TonemapParams(gamma: registry.float("Tonemap.Gamma"), mode: Int32(mode.rawValue))
        var rpDesc = RenderPassDescriptor()
        rpDesc.setName(name: "Tonemap")
        
        let upscalerType = registry.enum("Upscaler.Type", as: UpscalerType.self, default: .None)
        
        if upscalerType == .None {
            rpDesc.addAttachment(texture: context.drawable.texture, shouldClear: false)
        } else {
            if let ldrTexture = ldrTexture {
                rpDesc.addAttachment(texture: ldrTexture.texture, shouldClear: false)
                context.resources.register(ldrTexture, for: "LDR")
            } else {
                rpDesc.addAttachment(texture: context.drawable.texture, shouldClear: false)
            }
        }

        let rp = context.cmdBuffer.beginRenderPass(descriptor: rpDesc)
        rp.consumerBarrier(before: .vertex, after: [.vertex, .fragment, .mesh, .object, .dispatch])
        rp.setPipeline(pipeline: self.pipeline)
        rp.setTexture(texture: forward, index: 0, stages: .fragment)
        rp.setBytes(allocator: context.allocator, index: 0, bytes: &params, size: MemoryLayout<TonemapParams>.size, stages: .fragment)
        rp.draw(primitiveType: .triangle, vertexCount: 3, vertexOffset: 0)
        rp.producerBarrier(before: [.dispatch, .vertex, .fragment], after: .fragment)
        rp.end()
    }
}
