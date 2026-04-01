//
//  TextureVisualizerPass.swift
//  Pocketcat
//
//  Created by Amélie Heinrich on 01/04/2026.
//

import Metal
import simd
internal import QuartzCore

enum VisualizerPosition: Int, CaseIterable {
    case fullScreen  = 0
    case topLeft     = 1
    case topRight    = 2
    case bottomLeft  = 3
    case bottomRight = 4
}

class TextureVisualizerPass: Pass {
    private unowned let registry: SettingsRegistry
    private var pipelineCache: [String: RenderPipeline] = [:]
    private var outputWidth: Int = 1
    private var outputHeight: Int = 1
    private var lastLabels: [String] = []

    init(registry: SettingsRegistry) {
        self.registry = registry
        registry.register(bool: "Debug.TextureVisualizer", label: "Texture Visualizer", default: false)
        registry.register(dynamicPicker: "Debug.SelectedVisualizer", label: "Visualizer")
        registry.register(enum: "Debug.VisualizerPosition", label: "Position", default: VisualizerPosition.bottomRight)
        super.init()
    }

    override func resize(renderWidth: Int, renderHeight: Int, outputWidth: Int, outputHeight: Int) {
        self.outputWidth = outputWidth
        self.outputHeight = outputHeight
    }

    override func render(context: FrameContext) {
        let entries = context.resources.getVisualizers()

        // Keep UI combo box in sync — dispatch to main thread only when labels change
        let labels = entries.map { $0.label }
        if labels != lastLabels {
            lastLabels = labels
            let reg = registry
            DispatchQueue.main.async {
                reg.updatePickerOptions("Debug.SelectedVisualizer", options: labels)
            }
        }

        guard registry.bool("Debug.TextureVisualizer"), !entries.isEmpty else { return }

        let selectedIdx = registry.pickerIndex("Debug.SelectedVisualizer")
        guard selectedIdx < entries.count else { return }
        let entry = entries[selectedIdx]

        let position = registry.enum("Debug.VisualizerPosition", as: VisualizerPosition.self, default: .bottomRight)
        var rect: SIMD4<Float>

        if position == .fullScreen {
            // Full NDC: x=-1, y=1, w=2, h=2
            rect = SIMD4<Float>(-1.0, 1.0, 2.0, 2.0)
        } else {
            // Corner thumbnail: 33% of output height, square
            let thumbPx = Float(outputHeight) / 3.0
            let padding: Float = 12.0
            let W = Float(outputWidth)
            let H = Float(outputHeight)

            let xPx: Float
            let yTopPx: Float
            switch position {
            case .topLeft:
                xPx    = padding
                yTopPx = padding
            case .topRight:
                xPx    = W - thumbPx - padding
                yTopPx = padding
            case .bottomLeft:
                xPx    = padding
                yTopPx = H - thumbPx - padding
            default: // .bottomRight
                xPx    = W - thumbPx - padding
                yTopPx = H - thumbPx - padding
            }

            rect = SIMD4<Float>(
                xPx / W * 2.0 - 1.0,
                1.0 - yTopPx / H * 2.0,
                thumbPx / W * 2.0,
                thumbPx / H * 2.0
            )
        }

        var rpDesc = RenderPassDescriptor()
        rpDesc.name = "Texture Visualizer"
        rpDesc.addAttachment(texture: context.drawable.texture, shouldClear: false)

        let rp = context.cmdBuffer.beginRenderPass(descriptor: rpDesc)
        rp.consumerBarrier(before: .vertex, after: [.vertex, .fragment, .dispatch])
        rp.setPipeline(pipeline: pipeline(for: entry.fragmentFunction))
        rp.setTexture(texture: entry.texture, index: 0, stages: .fragment)
        rp.setBytes(allocator: context.allocator, index: 0,
                    bytes: &rect, size: MemoryLayout<SIMD4<Float>>.size, stages: .vertex)
        rp.draw(primitiveType: .triangle, vertexCount: 6, vertexOffset: 0)
        rp.end()
    }

    private func pipeline(for fragmentFunction: String) -> RenderPipeline {
        if let cached = pipelineCache[fragmentFunction] { return cached }
        var desc = RenderPipelineDescriptor()
        desc.name = "TexViz[\(fragmentFunction)]"
        desc.vertexFunction = "texviz_vs"
        desc.fragmentFunction = fragmentFunction
        desc.pixelFormats = [.bgra8Unorm]
        let p = RenderPipeline(descriptor: desc)
        pipelineCache[fragmentFunction] = p
        return p
    }
}
