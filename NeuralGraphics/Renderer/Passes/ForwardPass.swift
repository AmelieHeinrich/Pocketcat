//
//  ForwardPass.swift
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 02/03/2026.
//

import Metal
import simd

// Must match ModelData in Model.metal
private struct ModelData {
    var camera:       simd_float4x4
    var vertexOffset: UInt32
}

class ForwardPass: Pass {
    private let pipeline:      RenderPipeline
    private var depthTexture:  Texture
    private let defaultAlbedo: Texture
    private let allocator:     GPULinearAllocator

    override init() {
        var pipelineDesc               = RenderPipelineDescriptor()
        pipelineDesc.name              = "Forward Pipeline"
        pipelineDesc.vertexFunction    = "triangle_vs"
        pipelineDesc.fragmentFunction  = "triangle_fs"
        pipelineDesc.pixelFormats.append(.bgra8Unorm)
        pipelineDesc.depthFormat       = .depth32Float
        pipelineDesc.depthEnabled      = true
        pipelineDesc.depthWriteEnabled = true
        self.pipeline = RenderPipeline(descriptor: pipelineDesc)

        // 1×1 white fallback albedo
        let whiteDesc             = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false)
        whiteDesc.storageMode     = .shared
        whiteDesc.usage           = .shaderRead
        let white                 = Texture(descriptor: whiteDesc)
        white.setLabel(name: "Default White Albedo")
        var px: UInt32            = 0xFFFFFFFF
        withUnsafePointer(to: &px) { white.uploadData(region: MTLRegionMake2D(0, 0, 1, 1), mip: 0, data: UnsafeRawPointer($0), bpp: 4) }
        self.defaultAlbedo        = white

        // Depth buffer — starts at 1×1, resized on the first resize() call
        let depthDesc             = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float, width: 1, height: 1, mipmapped: false)
        depthDesc.usage           = .renderTarget
        depthDesc.storageMode     = .private
        let depth                 = Texture(descriptor: depthDesc)
        depth.setLabel(name: "Forward Depth Buffer")
        self.depthTexture         = depth

        // 16 MB per-frame scratch (model-data structs, one per instance)
        self.allocator            = GPULinearAllocator(size: 16 * 1024 * 1024)

        super.init()
    }

    override func resize(width: Int, height: Int) {
        depthTexture.resize(width: width, height: height)
    }

    override func render(context: FrameContext) {
        allocator.reset()

        var rpDesc = RenderPassDescriptor()
        rpDesc.addAttachment(texture: context.drawable.texture)
        rpDesc.setDepthAttachment(texture: depthTexture)
        rpDesc.name = "Forward Pass"

        let rp = context.cmdBuffer.beginRenderPass(descriptor: rpDesc)

        if let model = context.model {
            rp.setPipeline(pipeline: pipeline)

            for instance in model.instances {
                let offset = allocator.allocate(size: MemoryLayout<ModelData>.size)
                var data   = ModelData(camera: context.camera.viewProjection, vertexOffset: instance.vertexOffset)
                withUnsafePointer(to: &data) {
                    allocator.writeData(data: UnsafeRawPointer($0), offset: offset, size: MemoryLayout<ModelData>.size)
                }

                rp.setBuffer(buf: allocator.buffer, index: 0, stages: .vertex, offset: offset)
                rp.setBuffer(buf: model.vertexBuffer, index: 1, stages: .vertex)
                let albedo = model.materials[Int(instance.materialIndex)].albedo ?? defaultAlbedo
                rp.setTexture(texture: albedo, index: 0, stages: .fragment)
                rp.drawIndexed(primitimeType: .triangle, buffer: model.indexBuffer,
                               indexCount: Int(instance.indexCount),
                               indexOffset: UInt64(instance.indexOffset))
            }
        }

        rp.end()

        // Publish outputs so downstream passes and controllers can consume them
        context.resources.register(depthTexture, for: "Forward.Depth")
    }
}
