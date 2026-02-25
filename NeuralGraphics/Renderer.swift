//
//  Renderer.swift
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 23/02/2026.
//

import Metal
import MetalKit
import SwiftUI

protocol MetalViewDelegate : MTKViewDelegate {
    @MainActor func configure(_ view: MTKView)
}

class Renderer : NSObject, MetalViewDelegate {
    let device: MTLDevice
    let maxFramesInFlight: UInt64 = 3
    
    private let commandQueue: MTL4CommandQueue
    private let residencySet: MTLResidencySet
    private let compiler: MTL4Compiler
    private let commandBuffers: [CommandBuffer]
    private let renderPipeline: RenderPipeline
    
    init(device: MTLDevice) {
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
        self.commandQueue.addResidencySet(residencySet)
        self.compiler = try! device.makeCompiler(descriptor: compilerDescriptor)
        
        RendererData.initialize(device: self.device, cmdQueue: self.commandQueue, residencySet: self.residencySet, compiler: self.compiler)
        self.commandBuffers = (0..<maxFramesInFlight).map { _ in
            CommandBuffer()
        }
        
        var pipelineDescriptor = RenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = "triangle_vs"
        pipelineDescriptor.fragmentFunction = "triangle_fs"
        pipelineDescriptor.pixelFormats.append(.bgra8Unorm)
        
        self.renderPipeline = RenderPipeline(descriptor: pipelineDescriptor)
    }
    
    func configure(_ view: MTKView) {
        view.device = device
        view.colorPixelFormat = .bgra8Unorm
        view.clearColor = .init(red: 1.0, green: 1.0, blue: 1.0, alpha: 1.0)
        view.sampleCount = 1
        view.delegate = self
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
    
    func draw(in view: MTKView) {
        let frameIndex = Int(RendererData.gpuTimeline.currentValue % maxFramesInFlight)
        let cmdBuffer = commandBuffers[frameIndex]
        RendererData.gpuTimeline.wait(value: cmdBuffer.lastSignaledValue)

        let drawable = view.currentDrawable!
        var renderPassDescriptor = RenderPassDescriptor()
        renderPassDescriptor.addAttachment(texture: drawable.texture)
        
        cmdBuffer.begin()
    
        let renderPass = cmdBuffer.beginRenderPass(descriptor: renderPassDescriptor)
        renderPass.setPipeline(pipeline: self.renderPipeline)
        renderPass.draw(primitiveType: .triangle, vertexCount: 3, vertexOffset: 0)
        renderPass.end()
        cmdBuffer.end()
        
        commandQueue.waitForDrawable(drawable)
        cmdBuffer.commit()
        commandQueue.signalDrawable(drawable)
        drawable.present()
                
        cmdBuffer.lastSignaledValue = RendererData.gpuTimeline.signal()
    }
}
