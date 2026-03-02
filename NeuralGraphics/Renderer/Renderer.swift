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

    let camera: Camera = Camera()

    private let commandQueue: MTL4CommandQueue
    private let residencySet: MTLResidencySet
    private let compiler: MTL4Compiler
    private let commandBuffers: [CommandBuffer]
    private let renderPipeline: RenderPipeline
    private let resourceManager: ResourceManager
    private let indexBuffer: Buffer
    private let texture: MTLTexture
    private let gpuAllocator: GPULinearAllocator
    private var lastFrameTime: CFAbsoluteTime = CFAbsoluteTimeGetCurrent()

    init(device: MTLDevice) {
        self.device = device
        self.resourceManager = ResourceManager(device: self.device)

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
        self.commandBuffers = (0..<maxFramesInFlight).map { i in
            {
                let cmdBuffer = CommandBuffer()
                cmdBuffer.setName(name: "Command Buffer " + String(i))
                return cmdBuffer
            }()
        }

        var pipelineDescriptor = RenderPipelineDescriptor()
        pipelineDescriptor.name = "Triangle Pipeline"
        pipelineDescriptor.vertexFunction = "triangle_vs"
        pipelineDescriptor.fragmentFunction = "triangle_fs"
        pipelineDescriptor.pixelFormats.append(.bgra8Unorm)

        self.renderPipeline = RenderPipeline(descriptor: pipelineDescriptor)

        let indices: [UInt32] = [0, 1, 3, 1, 2, 3]
        self.indexBuffer = Buffer(bytes: indices, size: indices.count * MemoryLayout<UInt32>.size)
        self.indexBuffer.setName(name: "Index Buffer")

        self.texture = try! self.resourceManager.texture(url: Bundle.main.url(forResource: "TestTexture", withExtension: "png")!, sRGB: true)
        self.texture.label = "TestTexture.png"

        // 16 megabytes
        self.gpuAllocator = GPULinearAllocator(size: 16 * 1024 * 1024)
    }

    func configure(_ view: MTKView) {
        view.device = device
        view.colorPixelFormat = .bgra8Unorm
        view.clearColor = .init(red: 1.0, green: 1.0, blue: 1.0, alpha: 1.0)
        view.sampleCount = 1
        view.delegate = self
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        camera.resize(width: Float(size.width), height: Float(size.height))
    }

    func draw(in view: MTKView) {
        let now = CFAbsoluteTimeGetCurrent()
        let dt  = Float(now - lastFrameTime)
        lastFrameTime = now

        camera.update(dt: dt)
        gpuAllocator.reset()

        let frameIndex = Int(RendererData.gpuTimeline.currentValue % maxFramesInFlight)
        let cmdBuffer = commandBuffers[frameIndex]
        RendererData.gpuTimeline.wait(value: cmdBuffer.lastSignaledValue)

        let drawable = view.currentDrawable!

        var renderPassDescriptor = RenderPassDescriptor()
        renderPassDescriptor.addAttachment(texture: drawable.texture)
        renderPassDescriptor.name = "Draw Quads"

        var cameraMatrix = camera.projectionMatrix * camera.viewMatrix
        let offset = gpuAllocator.allocate(size: MemoryLayout<simd_float4x4>.size)
        withUnsafePointer(to: &cameraMatrix) { ptr in
            let ptr = UnsafeRawPointer(ptr)
            gpuAllocator.writeData(data: ptr, offset: offset, size: MemoryLayout<simd_float4x4>.size)
        }

        cmdBuffer.begin()

        let renderPass = cmdBuffer.beginRenderPass(descriptor: renderPassDescriptor)
        renderPass.setPipeline(pipeline: self.renderPipeline)
        renderPass.setBuffer(buf: gpuAllocator.buffer, index: 0, stages: .vertex, offset: offset)
        renderPass.setTexture(texture: self.texture, index: 0, stages: .fragment)
        renderPass.drawIndexed(primitimeType: .triangle, buffer: self.indexBuffer, indexCount: 6, indexOffset: 0)
        renderPass.end()
        cmdBuffer.end()

        commandQueue.waitForDrawable(drawable)
        cmdBuffer.commit()
        commandQueue.signalDrawable(drawable)
        drawable.present()

        cmdBuffer.lastSignaledValue = RendererData.gpuTimeline.signal()

        Input.shared.beginFrame()
    }
}
