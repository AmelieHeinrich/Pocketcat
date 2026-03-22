//
//  FrameManager.swift
//  Pocketcat
//
//  Created by Amélie Heinrich on 02/03/2026.
//

import Metal
internal import QuartzCore

// FrameManager is the top-level owner of the render graph.
//
//  - It *owns* the passes (strong references, lifetime management).
//  - It holds a RenderTimeline (weak handles to passes, swappable).
//  - It holds a RendererController (swappable strategy for driving a frame).
//  - It owns the command buffer ring and handles CPU/GPU synchronisation.
//
// Render flow:
//   FrameManager.render(drawable:)
//     → controller.render(timeline, &context)
//       → context.camera populated, timeline.execute(context)
//         → pass.render(context) for each pass in order

class FrameManager {
    private var frameIndex: UInt64 = 0
    private let maxFramesInFlight: Int = 3
    private var passes: [Pass]? = nil
    private var controller: RendererController
    private let commandBuffers: [CommandBuffer]
    private let resources: ResourceRegistry = ResourceRegistry()
    private let allocators: [GPULinearAllocator]
    private unowned let registry: SettingsRegistry

    private var viewportWidth: Int = 1
    private var viewportHeight: Int = 1
    private var lastRenderScale: Float = -1.0

    private var sceneBuffer: SceneBufferBuilder
    private var mobileTimeline: RenderTimeline? = nil
    private var desktopTimeline: RenderTimeline? = nil
    private var pathtracedTimeline: RenderTimeline? = nil

    var scene: RenderScene?

    init(registry: SettingsRegistry) {
        self.registry = registry
        self.controller = EditorRendererController(registry: registry)

        self.commandBuffers = (0..<3).map { i in
            let cb = CommandBuffer()
            cb.setName(name: "Command Buffer \(i)")
            return cb
        }
        self.allocators = (0..<3).map { i in
            let a = GPULinearAllocator(size: 32 * 1024 * 1024)
            return a
        }
        self.sceneBuffer = SceneBufferBuilder()

        setupTimelines(registry: registry)
    }

    func setScene(_ scene: RenderScene) {
        self.scene = scene
        self.sceneBuffer.build(scene: scene)
        RendererData.residencySet.commit()

        RendererData.waitIdle()
    }

    func resize(width: Int, height: Int) {
        viewportWidth = width
        viewportHeight = height
        let scale = registry.float("Renderer.RenderScale", default: 1.0)
        lastRenderScale = scale
        let scaledW = max(1, Int(Float(width) * scale))
        let scaledH = max(1, Int(Float(height) * scale))
        controller.resize(width: width, height: height)
        for pass in passes! {
            pass.resize(
                renderWidth: scaledW, renderHeight: scaledH, outputWidth: width,
                outputHeight: height)
        }
    }

    func render(drawable: CAMetalDrawable) {
        let currentScale = registry.float("Renderer.RenderScale", default: 1.0)
        if currentScale != lastRenderScale {
            lastRenderScale = currentScale
            let scaledW = max(1, Int(Float(viewportWidth) * currentScale))
            let scaledH = max(1, Int(Float(viewportHeight) * currentScale))
            for pass in passes! {
                pass.resize(
                    renderWidth: scaledW, renderHeight: scaledH, outputWidth: viewportWidth,
                    outputHeight: viewportHeight)
            }
        }

        resources.clear()

        let ringIndex = Int(frameIndex) % maxFramesInFlight
        let cmdBuffer = commandBuffers[ringIndex]
        let allocator = allocators[ringIndex]

        // Wait for cmdBuffer to be ready
        RendererData.gpuTimeline.wait(value: frameIndex)

        // Reset, record
        allocator.reset()
        cmdBuffer.begin()

        var context = FrameContext(
            camera: CameraData(),
            cmdBuffer: cmdBuffer,
            drawable: drawable,
            resources: resources,
            scene: scene,
            sceneBuffer: sceneBuffer,
            frameIndex: ringIndex,
            allocator: allocator
        )

        // Update entity transforms before passes run
        if let scene = scene {
            for (i, entity) in scene.entities.enumerated() {
                sceneBuffer.updateEntityTransform(i, transform: entity.transform)
            }
        }

        switch registry.enum("Renderer.Timeline", as: RendererTimelineType.self, default: .Desktop)
        {
        case .Mobile:
            controller.render(timeline: mobileTimeline!, context: &context)
        case .Desktop:
            controller.render(timeline: desktopTimeline!, context: &context)
        case .Pathtraced:
            controller.render(timeline: pathtracedTimeline!, context: &context)
        }

        // Commit
        cmdBuffer.end()
        cmdBuffer.commit()

        RendererData.cmdQueue.signalEvent(RendererData.gpuTimeline.event, value: frameIndex + 1)
        RendererData.cmdQueue.waitForEvent(RendererData.gpuTimeline.event, value: frameIndex + 1)
        frameIndex += 1

        RendererData.cmdQueue.waitForDrawable(drawable)
        RendererData.cmdQueue.signalDrawable(drawable)
        drawable.present()

        Input.shared.beginFrame()
    }

    func setupTimelines(registry: SettingsRegistry) {
        // Register global settings first so they appear at the top of the UI
        registry.register(
            enum: "Renderer.Timeline", label: "Timeline", default: RendererTimelineType.Desktop)
        registry.register(
            float: "Renderer.RenderScale", label: "Render Scale", default: 0.5, range: 0.25...1.0,
            step: 0.05)

        // Initialize passes
        let cullViewPass = CullViewPass(registry: registry)
        let visibilityPass = VisibilityBufferPass(registry: registry)
        let gbufferPass = GBufferPass()
        let tonemap = TonemapPass(registry: registry)
        let debug = DebugPass.shared
        let upscaler = MetalFXUpscalePass(registry: registry)
        let tlas = TLASBuildPass()
        let pathtracer = Pathtracer()
        let deferred = DeferredPass()
        registry.register(bool: "Debug.DepthTest", label: "Depth Test", default: false)
        debug.registry = registry

        self.passes = [
            tlas, cullViewPass, visibilityPass, pathtracer, tonemap, upscaler, debug, gbufferPass, deferred,
        ]

        // Desktop pipeline
        let desktopTimeline = RenderTimeline()
        desktopTimeline.addPass(tlas)
        desktopTimeline.addPass(cullViewPass)
        desktopTimeline.addPass(visibilityPass)
        desktopTimeline.addPass(gbufferPass)
        desktopTimeline.addPass(deferred)
        desktopTimeline.addPass(tonemap)
        desktopTimeline.addPass(upscaler)
        desktopTimeline.addPass(debug)

        // Pathtrace pipeline
        let pathtraceTimeline = RenderTimeline()
        pathtraceTimeline.addPass(tlas)
        pathtraceTimeline.addPass(pathtracer)
        pathtraceTimeline.addPass(tonemap)
        pathtraceTimeline.addPass(upscaler)

        self.desktopTimeline = desktopTimeline
        self.pathtracedTimeline = pathtraceTimeline
    }
}
