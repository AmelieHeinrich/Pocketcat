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
    private var frameIndex: UInt64 = 0
    private let maxFramesInFlight: Int = 3
    private var passes: [Pass]? = nil
    private var controller: RendererController
    private let commandBuffers: [CommandBuffer]
    private let resources: ResourceRegistry = ResourceRegistry()
    private let allocators: [GPULinearAllocator]
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
        self.allocators = (0..<3).map { i in
            let a = GPULinearAllocator(size: 32 * 1024 * 1024)
            return a
        }
        self.sceneBuffer = SceneBufferBuilder()

        setupTimelines(settings: settings)
    }

    func setScene(_ scene: RenderScene) {
        self.scene = scene
        self.sceneBuffer.build(scene: scene)
        RendererData.residencySet.commit()

        RendererData.waitIdle()
    }

    func resize(width: Int, height: Int) {
        controller.resize(width: width, height: height)
        for pass in passes! {
            pass.resize(width: width, height: height)
        }
    }

    func render(drawable: CAMetalDrawable) {
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

        switch settings.currentTimeline {
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

    func setupTimelines(settings: RendererSettings) {
        // Initialize passes
        let forward = ForwardPass(settings: settings)
        let tonemap = TonemapPass(settings: settings)
        let debug = DebugPass.shared
        let tlas = TLASBuildPass()
        debug.settings = settings

        self.passes = [tlas, forward, tonemap, debug]

        // Desktop pipeline
        let desktopTimeline = RenderTimeline()
        desktopTimeline.addPass(tlas)
        desktopTimeline.addPass(forward)
        desktopTimeline.addPass(tonemap)
        desktopTimeline.addPass(debug)

        self.desktopTimeline = desktopTimeline
    }
}
