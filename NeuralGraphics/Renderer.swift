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
    private let commandBuffer: MTL4CommandBuffer
    private let commandAllocators: [MTL4CommandAllocator]
    private let frameCompletionEvent: MTLSharedEvent
    
    private var frameIndex: UInt64 = 0
    
    init(device: MTLDevice) {
        self.device = device
        
        let cmdQueueDescriptor = MTL4CommandQueueDescriptor()
        cmdQueueDescriptor.label = "Main Graphics Queue"
        
        let residencySetDescriptor = MTLResidencySetDescriptor()
        residencySetDescriptor.initialCapacity = 16
        
        self.commandQueue = try! device.makeMTL4CommandQueue(descriptor: cmdQueueDescriptor)
        self.residencySet = try! device.makeResidencySet(descriptor: residencySetDescriptor)
        self.commandQueue.addResidencySet(residencySet)
        
        self.commandBuffer = device.makeCommandBuffer()!
        self.commandAllocators = (0..<maxFramesInFlight).map { _ in
            device.makeCommandAllocator()!
        }
        self.frameCompletionEvent = device.makeSharedEvent()!
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
        guard let renderPassDescriptor = view.currentMTL4RenderPassDescriptor else {
            return
        }
        
        if frameIndex >= maxFramesInFlight {
            let valueToWait = frameIndex - maxFramesInFlight
            frameCompletionEvent.wait(untilSignaledValue: valueToWait, timeoutMS: 8)
        }
        
        frameIndex += 1
        let allocator = commandAllocators[Int(frameIndex % maxFramesInFlight)]
        allocator.reset()
        
        commandBuffer.beginCommandBuffer(allocator: allocator)
        let commandEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!
        commandEncoder.endEncoding()
        commandBuffer.endCommandBuffer()
        
        let drawable = view.currentDrawable!
        
        commandQueue.waitForDrawable(drawable)
        commandQueue.commit([commandBuffer])
        commandQueue.signalDrawable(drawable)
        drawable.present()
                
        let valueToSignal = frameIndex
        commandQueue.signalEvent(frameCompletionEvent, value: valueToSignal)
    }
}
