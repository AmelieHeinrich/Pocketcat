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
        camera.applyJitter = (upscalerType == .Temporal || upscalerType == .TemporalDenoised)

        // The temporal denoised upscaler consumes RTGI/RTReflections as the diffuse/specular
        // albedo, so all RT effects must be enabled at Native tracing resolution to exist and
        // match the denoiser's input resolution.
        if upscalerType == .TemporalDenoised {
            registry.set(bool: "RTAO.Enabled", true)
            registry.set(bool: "RTGI.Enabled", true)
            registry.set(bool: "RTReflections.Enabled", true)
            registry.set(pickerIndex: "RTGI.TracingResolution", TracingResolution.Native.rawValue)
            registry.set(pickerIndex: "RTReflections.TracingResolution", TracingResolution.Native.rawValue)
        }

        camera.update(dt: dt)
        context.camera = camera.makeCameraData()
        context.sceneBuffer.updateCamera(context.camera)

        timeline.execute(context: context)
    }
}
