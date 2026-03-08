//
//  ForwardPass.swift
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 02/03/2026.
//

import Metal
internal import QuartzCore
import simd

// Must match ForwardPushConstants in Model.metal / ModelMS.metal
private struct ForwardPushConstants {
    var instanceIndex: UInt32
    var lod: UInt32
}

class ForwardPass: Pass {
    private let pipeline: RenderPipeline
    private let meshPipeline: MeshPipeline
    private var depthTexture: Texture
    private var colorTexture: Texture
    private unowned let settings: RendererSettings

    init(settings: RendererSettings) {
        self.settings = settings

        var meshPipelineDesc = MeshPipelineDescriptor()
        meshPipelineDesc.name = "Forward Pipeline (Mesh)"
        meshPipelineDesc.objectFunction = "forward_os"
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
        let sb = context.sceneBuffer

        guard sb.buffer != nil, sb.instanceCount > 0 else {
            rp.producerBarrier(before: .dispatch, after: .fragment)
            rp.end()
            context.resources.register(colorTexture, for: "Forward.Color")
            context.resources.register(depthTexture, for: "Forward.Depth")
            return
        }

        if settings.useMeshShader {
            rp.setMeshPipeline(pipeline: meshPipeline)

            // Bind scene buffer once for mesh + object + fragment stages
            rp.setBuffer(buf: sb.buffer, index: 0, stages: [.mesh, .object, .fragment])

            var globalInstanceIdx: UInt32 = 0
            if let scene = context.scene {
                for entity in scene.entities {
                    let model = entity.mesh
                    let lod = min(settings.forcedLOD, model.lodCount - 1)

                    for instance in model.instances {
                        var push = ForwardPushConstants(
                            instanceIndex: globalInstanceIdx,
                            lod: UInt32(lod))

                        let tgCountX = Int(instance.meshletCount[lod] / 32) + 1

                        rp.setBytes(
                            allocator: context.allocator, index: 1, bytes: &push,
                            size: MemoryLayout<ForwardPushConstants>.size, stages: .mesh)
                        rp.dispatchMesh(
                            threadgroupsPerGrid: MTLSizeMake(tgCountX, 1, 1),
                            threadsPerObjectThreadgroup: MTLSizeMake(32, 1, 1),
                            threadsPerMeshThreadgroup: MTLSizeMake(128, 1, 1))

                        globalInstanceIdx += 1
                    }
                }
            }
        } else {
            rp.setPipeline(pipeline: pipeline)

            // Bind scene buffer once for vertex + fragment stages
            rp.setBuffer(buf: sb.buffer, index: 0, stages: [.vertex, .fragment])

            var globalInstanceIdx: UInt32 = 0
            if let scene = context.scene {
                for entity in scene.entities {
                    let model = entity.mesh
                    let lod = min(settings.forcedLOD, model.lodCount - 1)
                    let lodData = model.lods[lod]

                    for instance in model.instances {
                        var push = ForwardPushConstants(
                            instanceIndex: globalInstanceIdx,
                            lod: UInt32(lod))

                        rp.setBytes(
                            allocator: context.allocator, index: 1, bytes: &push,
                            size: MemoryLayout<ForwardPushConstants>.size, stages: .vertex)
                        rp.drawIndexed(
                            primitimeType: .triangle, buffer: lodData.indexBuffer,
                            indexCount: Int(instance.indexCount[lod]),
                            indexOffset: UInt64(instance.indexOffset[lod]))

                        globalInstanceIdx += 1
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
