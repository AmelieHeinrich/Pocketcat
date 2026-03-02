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

struct ModelData {
    var camera: simd_float4x4
    var vertexOffset: uint
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
    private let gpuAllocator: GPULinearAllocator
    private var model: Mesh? = nil
    private var depthTexture: Texture
    private let defaultAlbedo: Texture
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
        pipelineDescriptor.depthFormat       = .depth32Float
        pipelineDescriptor.depthEnabled      = true
        pipelineDescriptor.depthWriteEnabled = true

        self.renderPipeline = RenderPipeline(descriptor: pipelineDescriptor)

        // 16 megabytes
        self.gpuAllocator = GPULinearAllocator(size: 16 * 1024 * 1024)

        // Depth buffer — starts at 1×1, resized on the first drawableSizeWillChange
        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: 1, height: 1, mipmapped: false)
        depthDesc.usage       = .renderTarget
        depthDesc.storageMode = .private
        self.depthTexture = Texture(descriptor: depthDesc)
        self.depthTexture.setLabel(name: "Depth Buffer")

        // 1×1 white fallback texture for materials with no albedo
        let whiteDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false)
        whiteDesc.storageMode = .shared
        whiteDesc.usage       = .shaderRead
        let defaultAlbedo = Texture(descriptor: whiteDesc)
        defaultAlbedo.setLabel(name: "Default White Albedo")
        var white: UInt32 = 0xFFFFFFFF
        withUnsafePointer(to: &white) { ptr in
            defaultAlbedo.uploadData(
                region: MTLRegionMake2D(0, 0, 1, 1), mip: 0,
                data: UnsafeRawPointer(ptr), bpp: 4)
        }
        self.defaultAlbedo = defaultAlbedo
    }

    /// Hands an already-loaded mesh to the renderer. Called on the main thread
    /// by ContentView once the background scene loader finishes.
    func setModel(_ mesh: Mesh) {
        self.model = mesh
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
        depthTexture.resize(width: Int(size.width), height: Int(size.height))
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
        renderPassDescriptor.setDepthAttachment(texture: depthTexture)
        renderPassDescriptor.name = "Draw Quads"

        cmdBuffer.begin()

        let renderPass = cmdBuffer.beginRenderPass(descriptor: renderPassDescriptor)
        if let model = model {
            renderPass.setPipeline(pipeline: self.renderPipeline)
            for instance in model.instances {
                let offset = gpuAllocator.allocate(size: MemoryLayout<ModelData>.size)
                var modelData = ModelData(camera: camera.projectionMatrix * camera.viewMatrix, vertexOffset: instance.vertexOffset)
                withUnsafePointer(to: &modelData) { ptr in
                    let ptr = UnsafeRawPointer(ptr)
                    gpuAllocator.writeData(data: ptr, offset: offset, size: MemoryLayout<ModelData>.size)
                }

                renderPass.setBuffer(buf: gpuAllocator.buffer, index: 0, stages: .vertex, offset: offset)
                renderPass.setBuffer(buf: model.vertexBuffer, index: 1, stages: .vertex)
                let albedo = model.materials[Int(instance.materialIndex)].albedo ?? defaultAlbedo
                renderPass.setTexture(texture: albedo, index: 0, stages: .fragment)
                renderPass.drawIndexed(primitimeType: .triangle, buffer: model.indexBuffer, indexCount: Int(instance.indexCount), indexOffset: UInt64(instance.indexOffset))
            }
        }
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
