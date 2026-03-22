//
//  EditorRendererController.swift
//  Pocketcat
//
//  Created by Amélie Heinrich on 02/03/2026.
//

import Foundation

class EditorRendererController: RendererController {
    let camera: Camera = Camera()
    unowned let registry: SettingsRegistry

    private var lastFrameTime: CFAbsoluteTime = CFAbsoluteTimeGetCurrent()

    init(registry: SettingsRegistry) {
        self.registry = registry
    }

    override func resize(width: Int, height: Int) {
        camera.resize(width: Float(width), height: Float(height))
    }

    override func render(timeline: RenderTimeline, context: inout FrameContext) {
        let now = CFAbsoluteTimeGetCurrent()
        let dt = Float(now - lastFrameTime)
        lastFrameTime = now

        let upscalerType = registry.enum("Upscaler.Type", as: UpscalerType.self, default: .Temporal)
        camera.applyJitter = (upscalerType == .Temporal)

        camera.update(dt: dt)
        context.camera = camera.makeCameraData()
        context.sceneBuffer.updateCamera(context.camera)

        timeline.execute(context: context)
    }
}
