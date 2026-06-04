import Metal
import simd

private struct ColorCorrectionParams {
    var temperature: Float
    var tint: Float
    var exposure: Float
    var contrast: Float
    var brightness: Float
    var saturation: Float
}

class ColorCorrectionPass: Pass {
    private let pipeline: ComputePipeline
    private unowned let registry: SettingsRegistry

    init(registry: SettingsRegistry) {
        self.registry = registry
        pipeline = ComputePipeline(function: "color_correction", name: "Color Correction")

        registry.register(float: "ColorCorrection.Temperature", label: "Temperature", default: 0.0,   range: -1000.0...1000.0)
        registry.register(float: "ColorCorrection.Tint",        label: "Tint",        default: 0.0,   range: -1000.0...1000.0)
        registry.register(float: "ColorCorrection.Exposure",    label: "Exposure",    default: 0.0,   range: -5.0...5.0)
        registry.register(float: "ColorCorrection.Contrast",    label: "Contrast",    default: 1.0,   range: 0.0...3.0)
        registry.register(float: "ColorCorrection.Brightness",  label: "Brightness",  default: 0.0,   range: -1.0...1.0)
        registry.register(float: "ColorCorrection.Saturation",  label: "Saturation",  default: 1.0,   range: 0.0...3.0)

        super.init()
    }

    override func render(context: FrameContext) {
        let hdr = context.resources.get("HDR") as Texture?
        guard let hdr else { return }

        var params = ColorCorrectionParams(
            temperature: registry.float("ColorCorrection.Temperature"),
            tint:        registry.float("ColorCorrection.Tint"),
            exposure:    registry.float("ColorCorrection.Exposure"),
            contrast:    registry.float("ColorCorrection.Contrast"),
            brightness:  registry.float("ColorCorrection.Brightness"),
            saturation:  registry.float("ColorCorrection.Saturation")
        )

        let w = hdr.texture.width
        let h = hdr.texture.height

        let cp = context.cmdBuffer.beginComputePass(name: "Color Correction")
        cp.consumerBarrier(before: .dispatch, after: .dispatch)
        cp.setPipeline(pipeline: pipeline)
        cp.setTexture(texture: hdr, index: 0)
        cp.setBytes(allocator: context.allocator, index: 0, bytes: &params, size: MemoryLayout<ColorCorrectionParams>.size)
        cp.dispatch(threads: MTLSizeMake((w + 7) / 8, (h + 7) / 8, 1), threadsPerGroup: MTLSizeMake(8, 8, 1))
        cp.end()
    }
}
