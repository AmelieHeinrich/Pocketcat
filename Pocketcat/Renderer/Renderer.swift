//
//  Renderer.swift
//  Pocketcat
//
//  Created by Amélie Heinrich on 23/02/2026.
//

import Metal
import MetalKit
import SwiftUI
import simd

protocol MetalViewDelegate: MTKViewDelegate {
    @MainActor func configure(_ view: MTKView)
}

class Renderer: NSObject, MetalViewDelegate {
    let device: MTLDevice

    private let commandQueue: MTL4CommandQueue
    private let residencySet: MTLResidencySet
    private let compiler: MTL4Compiler
    private let frameManager: FrameManager

    init(device: MTLDevice, registry: SettingsRegistry, lightState: LightState) {
        self.device = device

        let cmdQueueDescriptor = MTL4CommandQueueDescriptor()
        cmdQueueDescriptor.label = "Main Graphics Queue"

        let residencySetDescriptor = MTLResidencySetDescriptor()
        residencySetDescriptor.initialCapacity = 16
        residencySetDescriptor.label = "Residency Set"

        let compilerDescriptor = MTL4CompilerDescriptor()
        compilerDescriptor.label = "Shader Compiler"

        self.commandQueue = try! device.makeMTL4CommandQueue(descriptor: cmdQueueDescriptor)
        self.residencySet = try! device.makeResidencySet(descriptor: residencySetDescriptor)
        self.compiler = try! device.makeCompiler(descriptor: compilerDescriptor)
        self.commandQueue.addResidencySet(residencySet)

        RendererData.initialize(
            device: self.device,
            cmdQueue: self.commandQueue,
            residencySet: self.residencySet,
            compiler: self.compiler)

        self.frameManager = FrameManager(registry: registry, lightState: lightState)
    }

    func setScene(_ scene: RenderScene) {
        frameManager.setScene(scene)
    }

    func configure(_ view: MTKView) {
        view.device = device
        view.colorPixelFormat = .bgra8Unorm
        view.clearColor = .init(red: 0.0, green: 0.0, blue: 0.0, alpha: 1.0)
        view.sampleCount = 1
        view.delegate = self
        view.framebufferOnly = false
        view.preferredFramesPerSecond = 120

        self.commandQueue.addResidencySet(view.residencySet)
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        frameManager.resize(width: Int(size.width), height: Int(size.height))
    }

    func draw(in view: MTKView) {
        autoreleasepool {
            guard let drawable = view.currentDrawable else { return }
            
            frameManager.render(drawable: drawable)
        }
    }
}
