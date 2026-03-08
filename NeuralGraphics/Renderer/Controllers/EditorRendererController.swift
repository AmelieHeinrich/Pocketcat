//
//  EditorRendererController.swift
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 02/03/2026.
//

import Foundation

class EditorRendererController: RendererController {
    let camera: Camera = Camera()

    private var lastFrameTime: CFAbsoluteTime = CFAbsoluteTimeGetCurrent()

    override func resize(width: Int, height: Int) {
        camera.resize(width: Float(width), height: Float(height))
    }

    override func render(timeline: RenderTimeline, context: inout FrameContext) {
        let now = CFAbsoluteTimeGetCurrent()
        let dt = Float(now - lastFrameTime)
        lastFrameTime = now

        camera.update(dt: dt)
        context.camera = camera.makeCameraData()
        context.sceneBuffer.updateCamera(context.camera)

        timeline.execute(context: context)
    }
}
