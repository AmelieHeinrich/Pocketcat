//
//  FrameManager.swift
//  NeuralGraphics
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
    private let maxFramesInFlight: UInt64 = 3
    private var passes: [Pass]? = nil
    private var controller: RendererController
    private let commandBuffers: [CommandBuffer]
    private let resources: ResourceRegistry = ResourceRegistry()
    private let allocator: GPULinearAllocator
    private let settings: RendererSettings

    private var sceneBuffer: SceneBufferBuilder
    private var mobileTimeline: RenderTimeline? = nil
    private var desktopTimeline: RenderTimeline? = nil
    private var pathtracedTimeline: RenderTimeline? = nil

    var scene: RenderScene?

    init(settings: RendererSettings) {
        self.settings = settings
        self.controller = EditorRendererController()

        self.commandBuffers = (0..<3).map { i in
            let cb = CommandBuffer()
            cb.setName(name: "Command Buffer \(i)")
            return cb
        }
        self.allocator = GPULinearAllocator(size: 32 * 1024 * 1024)
        self.sceneBuffer = SceneBufferBuilder()

        setupTimelines(settings: settings)
    }

    func setScene(_ scene: RenderScene) {
        self.scene = scene
        self.sceneBuffer.build(scene: scene)
    }

    func resize(width: Int, height: Int) {
        controller.resize(width: width, height: height)
        for pass in passes! {
            pass.resize(width: width, height: height)
        }
    }

    func render(drawable: CAMetalDrawable) {
        resources.clear()

        let frameIndex = Int(RendererData.gpuTimeline.currentValue % maxFramesInFlight)
        let cmdBuffer = commandBuffers[frameIndex]
        RendererData.gpuTimeline.wait(value: cmdBuffer.lastSignaledValue)

        allocator.reset()
        cmdBuffer.begin()
        cmdBuffer.useResidencySet(drawable.layer.residencySet)
        cmdBuffer.useResidencySet(RendererData.residencySet)

        var context = FrameContext(
            camera: CameraData(),
            cmdBuffer: cmdBuffer,
            drawable: drawable,
            resources: resources,
            scene: scene,
            sceneBuffer: sceneBuffer,
            frameIndex: frameIndex,
            allocator: allocator
        )

        // Update entity transforms before passes run
        if let scene = scene {
            for (i, entity) in scene.entities.enumerated() {
                sceneBuffer.updateEntityTransform(i, transform: entity.transform)
            }
        }

        // The controller populates context.camera, syncs the scene buffer camera
        // (via context.sceneBuffer.updateCamera), then executes the timeline.
        switch settings.currentTimeline {
        case .Mobile:
            controller.render(timeline: mobileTimeline!, context: &context)
        case .Desktop:
            controller.render(timeline: desktopTimeline!, context: &context)
        case .Pathtraced:
            controller.render(timeline: pathtracedTimeline!, context: &context)
        }

        cmdBuffer.end()

        RendererData.cmdQueue.waitForDrawable(drawable)
        cmdBuffer.commit()
        RendererData.cmdQueue.signalDrawable(drawable)
        drawable.present()

        cmdBuffer.lastSignaledValue = RendererData.gpuTimeline.signal()

        Input.shared.beginFrame()
    }

    func setupTimelines(settings: RendererSettings) {
        // Initialize passes
        let forward = ForwardPass(settings: settings)
        let tonemap = TonemapPass(settings: settings)
        let debug = DebugPass.shared
        debug.settings = settings
        self.passes = [forward, tonemap, debug]

        // Desktop pipeline
        let desktopTimeline = RenderTimeline()
        desktopTimeline.addPass(forward)
        desktopTimeline.addPass(tonemap)
        desktopTimeline.addPass(debug)

        self.desktopTimeline = desktopTimeline
    }
}
