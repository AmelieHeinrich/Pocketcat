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

enum CubemapVisualizerShape: Int, CaseIterable {
    case sphere = 0
    case cube   = 1
}

private struct CubemapVizUniforms {
    var mvp: simd_float4x4
}

class TextureVisualizerPass: Pass {
    private unowned let registry: SettingsRegistry
    private var pipelineCache: [String: RenderPipeline] = [:]
    private var cubemapPipelineCache: [String: RenderPipeline] = [:]
    private var outputWidth: Int = 1
    private var outputHeight: Int = 1
    private var lastLabels: [String] = []
    private var lastCubemapLabels: [String] = []

    private var sphereVertexBuffer: Buffer?
    private var sphereVertexCount: Int = 0
    private var cubeVertexBuffer: Buffer?
    private var depthTexture: Texture?

    init(registry: SettingsRegistry) {
        self.registry = registry
        registry.register(bool: "Debug.TextureVisualizer", label: "Texture Visualizer", default: false)
        registry.register(dynamicPicker: "Debug.SelectedVisualizer", label: "Visualizer")
        registry.register(enum: "Debug.VisualizerPosition", label: "Position", default: VisualizerPosition.bottomRight)
        registry.register(bool: "Debug.CubemapVisualizer", label: "Cubemap Visualizer", default: false)
        registry.register(dynamicPicker: "Debug.SelectedCubemapViz", label: "Cubemap")
        registry.register(enum: "Debug.CubemapShape", label: "Shape", default: CubemapVisualizerShape.sphere)
        registry.register(float: "Debug.CubemapYaw", label: "Yaw", default: 0.0, range: -180.0...180.0)
        registry.register(float: "Debug.CubemapPitch", label: "Pitch", default: 0.0, range: -89.0...89.0)
        super.init()
        buildGeometry()
    }

    private func buildGeometry() {
        // --- Sphere ---
        let rings = 32, slices = 32
        var sphereVerts: [SIMD3<Float>] = []
        sphereVerts.reserveCapacity(rings * slices * 6)
        for ring in 0..<rings {
            let phi0 = Float.pi * Float(ring)     / Float(rings)
            let phi1 = Float.pi * Float(ring + 1) / Float(rings)
            for slice in 0..<slices {
                let theta0 = 2.0 * Float.pi * Float(slice)     / Float(slices)
                let theta1 = 2.0 * Float.pi * Float(slice + 1) / Float(slices)
                func p(_ phi: Float, _ theta: Float) -> SIMD3<Float> {
                    SIMD3(sin(phi) * cos(theta), cos(phi), sin(phi) * sin(theta))
                }
                let v00 = p(phi0, theta0), v10 = p(phi1, theta0)
                let v01 = p(phi0, theta1), v11 = p(phi1, theta1)
                sphereVerts += [v00, v10, v01, v10, v11, v01]
            }
        }
        sphereVertexCount = sphereVerts.count
        sphereVerts.withUnsafeBytes { ptr in
            sphereVertexBuffer = Buffer(bytes: ptr.baseAddress!, size: ptr.count)
        }
        sphereVertexBuffer?.setName(name: "Sphere Vertices (TexViz)")

        // --- Cube ---
        let cubeVerts: [SIMD3<Float>] = [
            // +Z
            SIMD3(-1,-1, 1), SIMD3( 1,-1, 1), SIMD3(-1, 1, 1),
            SIMD3( 1,-1, 1), SIMD3( 1, 1, 1), SIMD3(-1, 1, 1),
            // -Z
            SIMD3( 1,-1,-1), SIMD3(-1,-1,-1), SIMD3( 1, 1,-1),
            SIMD3(-1,-1,-1), SIMD3(-1, 1,-1), SIMD3( 1, 1,-1),
            // +X
            SIMD3( 1,-1, 1), SIMD3( 1,-1,-1), SIMD3( 1, 1, 1),
            SIMD3( 1,-1,-1), SIMD3( 1, 1,-1), SIMD3( 1, 1, 1),
            // -X
            SIMD3(-1,-1,-1), SIMD3(-1,-1, 1), SIMD3(-1, 1,-1),
            SIMD3(-1,-1, 1), SIMD3(-1, 1, 1), SIMD3(-1, 1,-1),
            // +Y
            SIMD3(-1, 1, 1), SIMD3( 1, 1, 1), SIMD3(-1, 1,-1),
            SIMD3( 1, 1, 1), SIMD3( 1, 1,-1), SIMD3(-1, 1,-1),
            // -Y
            SIMD3(-1,-1,-1), SIMD3( 1,-1,-1), SIMD3(-1,-1, 1),
            SIMD3( 1,-1,-1), SIMD3( 1,-1, 1), SIMD3(-1,-1, 1),
        ]
        cubeVerts.withUnsafeBytes { ptr in
            cubeVertexBuffer = Buffer(bytes: ptr.baseAddress!, size: ptr.count)
        }
        cubeVertexBuffer?.setName(name: "Cube Vertices (TexViz)")
    }

    override func resize(renderWidth: Int, renderHeight: Int, outputWidth: Int, outputHeight: Int) {
        self.outputWidth = outputWidth
        self.outputHeight = outputHeight
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: outputWidth, height: outputHeight, mipmapped: false)
        desc.usage = .renderTarget
        desc.storageMode = .private
        depthTexture = Texture(descriptor: desc)
        depthTexture?.setLabel(name: "Cubemap Visualizer Depth")
    }

    override func render(context: FrameContext) {
        // --- 2D Texture section ---
        let entries = context.resources.getVisualizers()
        let labels = entries.map { $0.label }
        if labels != lastLabels {
            lastLabels = labels
            let reg = registry
            DispatchQueue.main.async {
                reg.updatePickerOptions("Debug.SelectedVisualizer", options: labels)
            }
        }

        if registry.bool("Debug.TextureVisualizer"), !entries.isEmpty {
            let selectedIdx = registry.pickerIndex("Debug.SelectedVisualizer")
            if selectedIdx < entries.count {
                let entry = entries[selectedIdx]
                let position = registry.`enum`("Debug.VisualizerPosition", as: VisualizerPosition.self, default: .bottomRight)
                var rect = thumbnailRect(for: position)

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
        }

        // --- Cubemap section ---
        let cubemapEntries = context.resources.getCubemapVisualizers()
        let cubemapLabels = cubemapEntries.map { $0.label }
        if cubemapLabels != lastCubemapLabels {
            lastCubemapLabels = cubemapLabels
            let reg = registry
            DispatchQueue.main.async {
                reg.updatePickerOptions("Debug.SelectedCubemapViz", options: cubemapLabels)
            }
        }

        guard registry.bool("Debug.CubemapVisualizer"),
              !cubemapEntries.isEmpty,
              let depth = depthTexture else { return }

        let cubemapIdx = registry.pickerIndex("Debug.SelectedCubemapViz")
        guard cubemapIdx < cubemapEntries.count else { return }
        let cubemapEntry = cubemapEntries[cubemapIdx]

        let position = registry.`enum`("Debug.VisualizerPosition", as: VisualizerPosition.self, default: .bottomRight)
        let rect = thumbnailRect(for: position)

        let xPx = (rect.x + 1.0) * 0.5 * Float(outputWidth)
        let yPx = (1.0 - rect.y) * 0.5 * Float(outputHeight)
        let wPx = rect.z * 0.5 * Float(outputWidth)
        let hPx = rect.w * 0.5 * Float(outputHeight)
        let viewport = MTLViewport(originX: Double(xPx), originY: Double(yPx),
                                   width: Double(wPx), height: Double(hPx),
                                   znear: 0.0, zfar: 1.0)

        let yaw   = Float.radians(registry.float("Debug.CubemapYaw"))
        let pitch = Float.radians(registry.float("Debug.CubemapPitch"))
        let camDist: Float = 2.0
        let eye = SIMD3<Float>(sin(yaw) * cos(pitch) * camDist,
                               sin(pitch) * camDist,
                               cos(yaw) * cos(pitch) * camDist)
        let view = simd_float4x4.lookAtRH(eye: eye, center: .zero, up: SIMD3(0, 1, 0))
        let proj = simd_float4x4.perspectiveRH(fovY: Float.radians(60), aspect: 1.0, near: 0.1, far: 10.0)
        var uniforms = CubemapVizUniforms(mvp: proj * view)

        let shape = registry.`enum`("Debug.CubemapShape", as: CubemapVisualizerShape.self, default: .sphere)
        guard let vertexBuffer = (shape == .sphere ? sphereVertexBuffer : cubeVertexBuffer) else { return }
        let vertexCount = shape == .sphere ? sphereVertexCount : 36

        var rpDesc = RenderPassDescriptor()
        rpDesc.name = "Cubemap Visualizer"
        rpDesc.addAttachment(texture: context.drawable.texture, shouldClear: false)
        rpDesc.setDepthAttachment(texture: depth, shouldClear: true, shouldStore: false)

        let rp = context.cmdBuffer.beginRenderPass(descriptor: rpDesc)
        rp.consumerBarrier(before: .vertex, after: [.vertex, .fragment, .dispatch])
        rp.setViewport(viewport)
        rp.setPipeline(pipeline: cubemapPipeline(for: cubemapEntry.fragmentFunction))
        rp.setBytes(allocator: context.allocator, index: 0,
                    bytes: &uniforms, size: MemoryLayout<CubemapVizUniforms>.size, stages: .vertex)
        rp.setBuffer(buf: vertexBuffer, index: 1, stages: .vertex)
        rp.setTexture(texture: cubemapEntry.texture, index: 0, stages: .fragment)
        rp.draw(primitiveType: .triangle, vertexCount: vertexCount, vertexOffset: 0)
        rp.end()
    }

    private func thumbnailRect(for position: VisualizerPosition) -> SIMD4<Float> {
        if position == .fullScreen {
            return SIMD4<Float>(-1.0, 1.0, 2.0, 2.0)
        }
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
        default:
            xPx    = W - thumbPx - padding
            yTopPx = H - thumbPx - padding
        }
        return SIMD4<Float>(
            xPx / W * 2.0 - 1.0,
            1.0 - yTopPx / H * 2.0,
            thumbPx / W * 2.0,
            thumbPx / H * 2.0
        )
    }

    private func pipeline(for fragmentFunction: String) -> RenderPipeline {
        if let cached = pipelineCache[fragmentFunction] { return cached }
        var desc = RenderPipelineDescriptor()
        desc.name = "TexViz[\(fragmentFunction)]"
        desc.vertexFunction = "texviz_vs"
        desc.fragmentFunction = fragmentFunction
        desc.pixelFormats = [RendererData.getPixelFormat()]
        let p = RenderPipeline(descriptor: desc)
        pipelineCache[fragmentFunction] = p
        return p
    }

    private func cubemapPipeline(for fragmentFunction: String) -> RenderPipeline {
        if let cached = cubemapPipelineCache[fragmentFunction] { return cached }
        var desc = RenderPipelineDescriptor()
        desc.name = "CubemapViz[\(fragmentFunction)]"
        desc.vertexFunction = "texviz_cubemap_vs"
        desc.fragmentFunction = fragmentFunction
        desc.pixelFormats = [RendererData.getPixelFormat()]
        desc.depthFormat = .depth32Float
        desc.depthEnabled = true
        desc.depthWriteEnabled = true
        let p = RenderPipeline(descriptor: desc)
        cubemapPipelineCache[fragmentFunction] = p
        return p
    }
}
