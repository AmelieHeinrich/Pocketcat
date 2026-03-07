//
//  ForwardPass.swift
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 02/03/2026.
//

import Metal
internal import QuartzCore
import simd

// Must match ModelData in Model.metal / ModelMS.metal
private struct ModelData {
    var camera: simd_float4x4
    var vertexOffset: UInt32
}

class ForwardPass: Pass {
    private let pipeline: RenderPipeline
    private let meshPipeline: MeshPipeline
    private var depthTexture: Texture
    private var colorTexture: Texture
    private let defaultAlbedo: Texture
    private unowned let settings: RendererSettings

    init(settings: RendererSettings) {
        self.settings = settings

        var meshPipelineDesc = MeshPipelineDescriptor()
        meshPipelineDesc.name = "Forward Pipeline (Mesh)"
        meshPipelineDesc.meshFunction = "forward_ms"
        meshPipelineDesc.fragmentFunction = "forward_msfs"
        meshPipelineDesc.pixelFormats.append(.rgba16Float)
        meshPipelineDesc.depthFormat = .depth32Float
        meshPipelineDesc.depthEnabled = true
        meshPipelineDesc.depthWriteEnabled = true
        self.meshPipeline = MeshPipeline(descriptor: meshPipelineDesc)

        var pipelineDesc = RenderPipelineDescriptor()
        pipelineDesc.name = "Forward Pipeline"
        pipelineDesc.vertexFunction = "forward_vs"
        pipelineDesc.fragmentFunction = "forward_vsfs"
        pipelineDesc.pixelFormats.append(.rgba16Float)
        pipelineDesc.depthFormat = .depth32Float
        pipelineDesc.depthEnabled = true
        pipelineDesc.depthWriteEnabled = true
        self.pipeline = RenderPipeline(descriptor: pipelineDesc)

        // 1×1 white fallback albedo
        let whiteDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false)
        whiteDesc.storageMode = .shared
        whiteDesc.usage = .shaderRead
        let white = Texture(descriptor: whiteDesc)
        white.setLabel(name: "Default White Albedo")
        var px: UInt32 = 0xFFFF_FFFF
        withUnsafePointer(to: &px) {
            white.uploadData(
                region: MTLRegionMake2D(0, 0, 1, 1), mip: 0, data: UnsafeRawPointer($0), bpp: 4)
        }
        self.defaultAlbedo = white

        // Color buffer
        let colorDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: 1, height: 1, mipmapped: false)
        colorDesc.usage = [.shaderRead, .renderTarget, .shaderWrite]
        let color = Texture(descriptor: colorDesc)
        color.setLabel(name: "Forward Color Buffer")
        self.colorTexture = color

        // Depth buffer
        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: 1, height: 1, mipmapped: false)
        depthDesc.usage = .renderTarget
        let depth = Texture(descriptor: depthDesc)
        depth.setLabel(name: "Forward Depth Buffer")
        self.depthTexture = depth

        super.init()
    }

    override func resize(width: Int, height: Int) {
        colorTexture.resize(width: width, height: height)
        depthTexture.resize(width: width, height: height)
    }

    override func render(context: FrameContext) {
        var rpDesc = RenderPassDescriptor()
        rpDesc.addAttachment(texture: self.colorTexture)
        rpDesc.setDepthAttachment(texture: depthTexture)
        rpDesc.name = "Forward Pass"

        let rp = context.cmdBuffer.beginRenderPass(descriptor: rpDesc)

        if let scene = context.scene {
            for entity in scene.entities {
                let model = entity.mesh
                if settings.useMeshShader {
                    rp.setMeshPipeline(pipeline: meshPipeline)

                    let lod = min(settings.forcedLOD, model.lodCount - 1)
                    let lodData = model.lods[lod]

                    for instance in model.instances {
                        let mvp = context.camera.projection * context.camera.view * entity.transform
                        var data = ModelData(camera: mvp, vertexOffset: instance.vertexOffset)

                        rp.setBytes(
                            allocator: context.allocator, index: 0, bytes: &data,
                            size: MemoryLayout<ModelData>.size, stages: .mesh)
                        rp.setBuffer(
                            buf: lodData.meshletBuffer, index: 1, stages: .mesh,
                            offset: Int(instance.meshletOffset[lod]) * 16)
                        rp.setBuffer(buf: lodData.meshletVerticesBuffer, index: 2, stages: .mesh)
                        rp.setBuffer(buf: lodData.meshletTrianglesBuffer, index: 3, stages: .mesh)
                        rp.dispatchMesh(
                            threadgroupsPerGrid: MTLSizeMake(Int(instance.meshletCount[lod]), 1, 1),
                            threadsPerObjectThreadgroup: MTLSizeMake(0, 0, 0),
                            threadsPerMeshThreadgroup: MTLSizeMake(128, 1, 1)
                        )
                    }
                } else {
                    rp.setPipeline(pipeline: pipeline)

                    let lod = min(settings.forcedLOD, model.lodCount - 1)
                    let lodData = model.lods[lod]

                    for instance in model.instances {
                        let mvp = context.camera.projection * context.camera.view * entity.transform
                        var data = ModelData(camera: mvp, vertexOffset: instance.vertexOffset)
                        let albedo =
                            model.materials.indices.contains(Int(instance.materialIndex))
                            ? (model.materials[Int(instance.materialIndex)].albedo ?? defaultAlbedo)
                            : defaultAlbedo

                        rp.setBytes(
                            allocator: context.allocator, index: 0, bytes: &data,
                            size: MemoryLayout<ModelData>.size, stages: .vertex)
                        rp.setBuffer(buf: model.vertexBuffer, index: 1, stages: .vertex)
                        rp.setTexture(texture: albedo, index: 0, stages: .fragment)
                        rp.drawIndexed(
                            primitimeType: .triangle, buffer: lodData.indexBuffer,
                            indexCount: Int(instance.indexCount[lod]),
                            indexOffset: UInt64(instance.indexOffset[lod]))
                    }
                }
            }
        }

        rp.producerBarrier(before: .dispatch, after: .fragment)
        rp.end()

        // Publish outputs so downstream passes and controllers can consume them
        context.resources.register(colorTexture, for: "Forward.Color")
        context.resources.register(depthTexture, for: "Forward.Depth")
    }
}
