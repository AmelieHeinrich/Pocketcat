//
//  ForwardPass.swift
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 02/03/2026.
//

import Metal
internal import QuartzCore
import simd

class ForwardPass: Pass {
    private let pipeline: RenderPipeline
    private let meshPipeline: MeshPipeline
    private let vertexCullPipe: ComputePipeline
    private let meshCullPipe: ComputePipeline
    private let icbResetPipe: ComputePipeline
    private let vertexICBs: [ICB]
    private let meshICBs: [ICB]
    private let instanceIDBuffers: [Buffer]

    private var depthTexture: Texture
    private var colorTexture: Texture
    private unowned let settings: RendererSettings

    init(settings: RendererSettings) {
        self.settings = settings

        self.icbResetPipe = ComputePipeline(function: "reset_icb", name: "Reset ICB")
        self.vertexCullPipe = ComputePipeline(
            function: "vertex_geometry_cull", name: "Cull Instances (VS)")
        self.meshCullPipe = ComputePipeline(
            function: "mesh_geometry_cull", name: "Cull Instances (MS)")

        self.vertexICBs = (0..<3).map { i in
            let icb = ICB(inherit: false, commandTypes: .drawIndexed, maxCommandCount: 65536)
            icb.setName(label: "Vertex Forward ICB \(i)")
            return icb
        }

        self.meshICBs = (0..<3).map { i in
            let icb = ICB(
                inherit: false, commandTypes: .drawMeshThreadgroups, maxCommandCount: 65536)
            icb.setName(label: "Mesh Forward ICB \(i)")
            return icb
        }

        var meshPipelineDesc = MeshPipelineDescriptor()
        meshPipelineDesc.name = "Forward Pipeline (Mesh)"
        meshPipelineDesc.objectFunction = "forward_os"
        meshPipelineDesc.meshFunction = "forward_ms"
        meshPipelineDesc.fragmentFunction = "forward_msfs"
        meshPipelineDesc.pixelFormats.append(.rgba16Float)
        meshPipelineDesc.depthFormat = .depth32Float
        meshPipelineDesc.depthEnabled = true
        meshPipelineDesc.depthWriteEnabled = true
        meshPipelineDesc.supportsIndirect = true
        meshPipelineDesc.depthCompareOp = .less
        self.meshPipeline = MeshPipeline(descriptor: meshPipelineDesc)

        var pipelineDesc = RenderPipelineDescriptor()
        pipelineDesc.name = "Forward Pipeline"
        pipelineDesc.vertexFunction = "forward_vs"
        pipelineDesc.fragmentFunction = "forward_vsfs"
        pipelineDesc.pixelFormats.append(.rgba16Float)
        pipelineDesc.depthCompareOp = .less
        pipelineDesc.depthFormat = .depth32Float
        pipelineDesc.depthEnabled = true
        pipelineDesc.depthWriteEnabled = true
        pipelineDesc.supportsIndirect = true
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

        self.instanceIDBuffers = (0..<3).map { i in
            let buffer = Buffer(size: 65536 * MemoryLayout<UInt32>.size)
            buffer.setName(name: "Instance ID Buffer \(i) (Mesh ICB)")
            return buffer
        }

        super.init()
    }

    override func resize(width: Int, height: Int) {
        colorTexture.resize(width: width, height: height)
        depthTexture.resize(width: width, height: height)
    }

    override func render(context: FrameContext) {
        let sb = context.sceneBuffer

        guard sb.buffer != nil, sb.instanceCount > 0 else {
            context.resources.register(colorTexture, for: "Forward.Color")
            context.resources.register(depthTexture, for: "Forward.Depth")
            return
        }
        if settings.useMeshShader {
            meshPathMTL4(context: context)
        } else {
            vertexPath(context: context)
        }

        // Publish outputs so downstream passes and controllers can consume them
        context.resources.register(colorTexture, for: "Forward.Color")
        context.resources.register(depthTexture, for: "Forward.Depth")
    }

    func vertexPath(context: FrameContext) {
        var maxInstanceCount = 65536
        var instanceCount = context.sceneBuffer.instanceCount
        let resetTgCountX = (Int(maxInstanceCount) + 63) / 64
        let cullTgCountX = (Int(instanceCount) + 63) / 64
        let vertexICB = vertexICBs[context.frameIndex]
        let instanceIDBuffer = instanceIDBuffers[context.frameIndex]

        var rpDesc = RenderPassDescriptor()
        rpDesc.addAttachment(texture: self.colorTexture)
        rpDesc.setDepthAttachment(texture: depthTexture)
        rpDesc.name = "Forward Pass"

        // Cull
        let cp = context.cmdBuffer.beginComputePass(name: "Reset & Cull Instances")
        cp.setPipeline(pipeline: icbResetPipe)
        cp.setBuffer(buf: vertexICB.buffer, index: 0)
        cp.setBytes(
            allocator: context.allocator, index: 1, bytes: &maxInstanceCount,
            size: MemoryLayout<Int>.size)
        cp.dispatch(
            threads: MTLSizeMake(resetTgCountX, 1, 1), threadsPerGroup: MTLSizeMake(64, 1, 1))
        cp.intraPassBarrier(before: .dispatch, after: .dispatch)
        cp.setPipeline(pipeline: vertexCullPipe)
        cp.setBuffer(buf: context.sceneBuffer.buffer, index: 0)
        cp.setBuffer(buf: vertexICB.buffer, index: 1)
        cp.setBytes(
            allocator: context.allocator, index: 2, bytes: &instanceCount,
            size: MemoryLayout<UInt32>.size)
        cp.dispatch(
            threads: MTLSizeMake(cullTgCountX, 1, 1), threadsPerGroup: MTLSizeMake(64, 1, 1))
        cp.end()

        // Flush
        let rp = context.cmdBuffer.beginRenderPass(descriptor: rpDesc)
        rp.consumerBarrier(before: .vertex, after: [.dispatch, .accelerationStructure])
        rp.setPipeline(pipeline: pipeline)
        rp.executeIndirect(icb: vertexICB, maxCommandCount: 65536)
        rp.end()
    }

    func meshPathMTL3(context: FrameContext) {
        let meshICB = meshICBs[context.frameIndex]
        let instanceIDBuffer = instanceIDBuffers[context.frameIndex]

        var instanceCount = context.sceneBuffer.instanceCount
        let cullTgCountX = (Int(instanceCount) + 63) / 64

        RendererData.mtl3commandBuffer = RendererData.mtl3commandQueue.makeCommandBuffer()
        RendererData.mtl3commandBuffer.useResidencySet(RendererData.residencySet)

        // Cull
        let bp = RendererData.mtl3commandBuffer.makeBlitCommandEncoder()!
        bp.resetCommandsInBuffer(meshICB.cmdBuffer, range: 0..<65536)
        bp.updateFence(RendererData.gpuTimeline.fence)
        bp.endEncoding()

        let cp = RendererData.mtl3commandBuffer.makeComputeCommandEncoder()!
        cp.waitForFence(RendererData.gpuTimeline.fence)
        cp.setComputePipelineState(meshCullPipe.pipelineState)
        cp.setBuffer(context.sceneBuffer.buffer.buffer, offset: 0, index: 0)
        cp.setBuffer(meshICB.buffer.buffer, offset: 0, index: 1)
        cp.setBytes(&instanceCount, length: MemoryLayout<UInt32>.size, index: 2)
        cp.setBuffer(instanceIDBuffer.buffer, offset: 0, index: 3)
        cp.dispatchThreadgroups(
            MTLSizeMake(cullTgCountX, 1, 1), threadsPerThreadgroup: MTLSizeMake(64, 1, 1))
        cp.updateFence(RendererData.gpuTimeline.fence)
        cp.endEncoding()

        // Flush
        let rpd = MTLRenderPassDescriptor()
        rpd.colorAttachments[0].texture = self.colorTexture.texture
        rpd.colorAttachments[0].clearColor = .init()
        rpd.colorAttachments[0].loadAction = .clear
        rpd.colorAttachments[0].storeAction = .store
        rpd.depthAttachment.texture = self.depthTexture.texture
        rpd.depthAttachment.clearDepth = 1.0
        rpd.depthAttachment.storeAction = .store
        rpd.depthAttachment.loadAction = .clear

        let rp = RendererData.mtl3commandBuffer.makeRenderCommandEncoder(descriptor: rpd)!
        rp.waitForFence(RendererData.gpuTimeline.fence, before: .object)
        rp.setRenderPipelineState(meshPipeline.pipelineState)
        rp.setDepthStencilState(meshPipeline.depthStencilState)
        rp.executeCommandsInBuffer(meshICB.cmdBuffer, range: 0..<65536)
        rp.updateFence(RendererData.gpuTimeline.fence, after: .fragment)
        rp.endEncoding()

        RendererData.mtl3commandBuffer.commit()
        RendererData.mtl3commandBuffer.waitUntilCompleted()
    }

    func meshPathMTL4(context: FrameContext) {
        let meshICB = meshICBs[context.frameIndex]
        let instanceIDBuffer = instanceIDBuffers[context.frameIndex]

        var maxInstanceCount = 65536
        var instanceCount = context.sceneBuffer.instanceCount
        let resetTgCountX = (Int(maxInstanceCount) + 63) / 64
        let cullTgCountX = (Int(instanceCount) + 63) / 64

        var rpDesc = RenderPassDescriptor()
        rpDesc.addAttachment(texture: self.colorTexture)
        rpDesc.setDepthAttachment(texture: depthTexture)
        rpDesc.name = "Forward Pass"

        let cp = context.cmdBuffer.beginComputePass(name: "Reset & Cull Instances")
        cp.setPipeline(pipeline: icbResetPipe)
        cp.setBuffer(buf: meshICB.buffer, index: 0)
        cp.setBytes(
            allocator: context.allocator, index: 1, bytes: &maxInstanceCount,
            size: MemoryLayout<Int>.size)
        cp.dispatch(
            threads: MTLSizeMake(resetTgCountX, 1, 1), threadsPerGroup: MTLSizeMake(64, 1, 1))
        cp.intraPassBarrier(before: .dispatch, after: .dispatch)
        cp.setPipeline(pipeline: meshCullPipe)
        cp.setBuffer(buf: context.sceneBuffer.buffer, index: 0)
        cp.setBuffer(buf: meshICB.buffer, index: 1)
        cp.setBytes(
            allocator: context.allocator, index: 2, bytes: &instanceCount,
            size: MemoryLayout<UInt32>.size)
        cp.setBuffer(buf: instanceIDBuffer, index: 3)
        cp.dispatch(
            threads: MTLSizeMake(cullTgCountX, 1, 1), threadsPerGroup: MTLSizeMake(64, 1, 1))
        cp.end()

        let rp = context.cmdBuffer.beginRenderPass(descriptor: rpDesc)
        rp.consumerBarrier(before: .object, after: [.dispatch, .accelerationStructure])
        rp.setMeshPipeline(pipeline: meshPipeline)
        rp.executeIndirect(icb: meshICB, maxCommandCount: 65536)
        //rp.setBuffer(buf: context.sceneBuffer.buffer, index: 0, stages: [.mesh, .object, .fragment])
        //var globalInstanceIdx: UInt32 = 0
        //if let scene = context.scene {
        //    for entity in scene.entities {
        //        let model = entity.mesh
        //
        //        for _ in model.instances {
        //            rp.setBytes(allocator: context.allocator, index: 1, bytes: &globalInstanceIdx, size: MemoryLayout<UInt32>.size, stages: .object)
        //            rp.dispatchMesh(
        //                threadgroupsPerGrid: MTLSizeMake(1, 1, 1),
        //                threadsPerObjectThreadgroup: MTLSizeMake(32, 1, 1),
        //                threadsPerMeshThreadgroup: MTLSizeMake(128, 1, 1))
        //
        //            globalInstanceIdx += 1
        //        }
        //    }
        //}
        rp.end()
    }
}
