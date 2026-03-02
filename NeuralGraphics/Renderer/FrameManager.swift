//
//  FrameManager.swift
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 02/03/2026.
//

import Metal

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

import QuartzCore

class FrameManager {
    private let maxFramesInFlight: UInt64 = 3

    // Owned passes
    private var passes: [Pass]

    // Swappable timeline + controller
    private var timeline:   RenderTimeline
    private var controller: RendererController

    // Frame pacing
    private let commandBuffers: [CommandBuffer]

    // Shared resource blackboard — cleared at the top of every frame
    private let resources: ResourceRegistry = ResourceRegistry()

    var model: Mesh?

    init() {
        // Build the forward pass and register it in the timeline
        let forward = ForwardPass()
        self.passes = [forward]

        let timeline = RenderTimeline()
        timeline.addPass(forward)
        self.timeline   = timeline

        self.controller = EditorRendererController()

        self.commandBuffers = (0..<3).map { i in
            let cb = CommandBuffer()
            cb.setName(name: "Command Buffer \(i)")
            return cb
        }
    }

    func setModel(_ mesh: Mesh) {
        model = mesh
    }

    func resize(width: Int, height: Int) {
        controller.resize(width: width, height: height)
        for pass in passes {
            pass.resize(width: width, height: height)
        }
    }

    func render(drawable: CAMetalDrawable) {
        resources.clear()

        let frameIndex = Int(RendererData.gpuTimeline.currentValue % maxFramesInFlight)
        let cmdBuffer  = commandBuffers[frameIndex]
        RendererData.gpuTimeline.wait(value: cmdBuffer.lastSignaledValue)

        cmdBuffer.begin()

        var context = FrameContext(
            camera:     CameraData(),
            cmdBuffer:  cmdBuffer,
            drawable:   drawable,
            resources:  resources,
            model:      model,
            frameIndex: frameIndex
        )

        controller.render(timeline: timeline, context: &context)

        cmdBuffer.end()

        RendererData.cmdQueue.waitForDrawable(drawable)
        cmdBuffer.commit()
        RendererData.cmdQueue.signalDrawable(drawable)
        drawable.present()

        cmdBuffer.lastSignaledValue = RendererData.gpuTimeline.signal()

        Input.shared.beginFrame()
    }
}
