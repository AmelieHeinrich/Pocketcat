//
//  MetalView.swift
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 22/02/2026.
//

import SwiftUI
import MetalKit

struct MetalView: NSViewRepresentable {
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeNSView(context: Context) -> MTKView {
        let view = MTKView()
        view.device = MTLCreateSystemDefaultDevice()
        view.clearColor = MTLClearColorMake(0.1, 0.1, 0.1, 1.0)
        view.delegate = context.coordinator
        return view
    }

    func updateNSView(_ nsView: MTKView, context: Context) {}

    class Coordinator: NSObject, MTKViewDelegate {
        nonisolated func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

        nonisolated func draw(in view: MTKView) {
            guard let drawable = view.currentDrawable,
                  let device = view.device,
                  let queue = device.makeCommandQueue(),
                  let buffer = queue.makeCommandBuffer()
            else { return }
            
            let descriptor = MTLRenderPassDescriptor()
            descriptor.colorAttachments[0].clearColor = MTLClearColorMake(0.1, 0.1, 0.5, 1.0)
            descriptor.colorAttachments[0].storeAction = .store
            descriptor.colorAttachments[0].loadAction = .clear
            descriptor.colorAttachments[0].texture = drawable.texture
            
            guard let encoder = buffer.makeRenderCommandEncoder(descriptor: descriptor)
            else { return }
            
            encoder.endEncoding()
            buffer.present(drawable)
            buffer.commit()
        }
    }
}
