//
//  RenderPass.swift
//  Pocketcat
//
//  Created by Amélie Heinrich on 25/02/2026.
//

import Metal
import simd

struct RenderPassDescriptor {
    var name: String = "Render Pass"
    var colorTextures: [MTLTexture] = []
    var clearColors: [simd_float4] = []
    var shouldClear: [Bool] = []

    var depthTexture: MTLTexture! = nil
    var shouldClearDepth: Bool = true
    var shouldStoreDepth: Bool = true

    mutating func setName(name: String) {
        self.name = name
    }

    mutating func addAttachment(
        texture: Texture, clearColor: simd_float4 = simd_float4(0.0, 0.0, 0.0, 1.0),
        shouldClear: Bool = true
    ) {
        self.colorTextures.append(texture.texture)
        self.clearColors.append(clearColor)
        self.shouldClear.append(shouldClear)
    }

    mutating func addAttachment(
        texture: MTLTexture, clearColor: simd_float4 = simd_float4(0.0, 0.0, 0.0, 1.0),
        shouldClear: Bool = true
    ) {
        self.colorTextures.append(texture)
        self.clearColors.append(clearColor)
        self.shouldClear.append(shouldClear)
    }

    mutating func setDepthAttachment(
        texture: Texture, shouldClear: Bool = true, shouldStore: Bool = true
    ) {
        self.depthTexture = texture.texture
        self.shouldClearDepth = shouldClear
        self.shouldStoreDepth = shouldStore
    }
}

class RenderPass {
    var descriptor: RenderPassDescriptor
    var encoder: MTL4RenderCommandEncoder

    init(descriptor: RenderPassDescriptor, cmdBuffer: MTL4CommandBuffer) {
        self.descriptor = descriptor

        let rpd = MTL4RenderPassDescriptor()
        for i in 0..<descriptor.colorTextures.count {
            rpd.colorAttachments[i].texture = descriptor.colorTextures[i]
            rpd.colorAttachments[i].clearColor = MTLClearColorMake(
                Double(descriptor.clearColors[i].x), Double(descriptor.clearColors[i].y),
                Double(descriptor.clearColors[i].z), Double(descriptor.clearColors[i].w))
            rpd.colorAttachments[i].storeAction = .store
            rpd.colorAttachments[i].loadAction = descriptor.shouldClear[i] ? .clear : .load
        }
        if descriptor.depthTexture != nil {
            rpd.depthAttachment.texture = descriptor.depthTexture
            rpd.depthAttachment.loadAction = descriptor.shouldClearDepth ? .clear : .load
            rpd.depthAttachment.clearDepth = 1.0
            rpd.depthAttachment.storeAction = descriptor.shouldStoreDepth ? .store : .dontCare
        }

        self.encoder = cmdBuffer.makeRenderCommandEncoder(descriptor: rpd)!
        self.encoder.label = descriptor.name

        if let heap = RendererData.counterHeap {
            let slot = RendererData.counterOffset
            RendererData.counterOffset += 1
            RendererData.counterEntries.append((name: descriptor.name, startSlot: slot, endSlot: -1))
            self.encoder.writeTimestamp(granularity: .relaxed, after: .fragment, counterHeap: heap, index: slot)
        }
    }

    func end() {
        if let heap = RendererData.counterHeap {
            let slot = RendererData.counterOffset
            RendererData.counterOffset += 1
            if !RendererData.counterEntries.isEmpty {
                RendererData.counterEntries[RendererData.counterEntries.count - 1].endSlot = slot
            }
            self.encoder.writeTimestamp(granularity: .relaxed, after: .fragment, counterHeap: heap, index: slot)
        }
        self.encoder.endEncoding()
    }

    func setPipeline(pipeline: RenderPipeline) {
        self.encoder.setRenderPipelineState(pipeline.pipelineState)
        if pipeline.depthStencilState != nil {
            self.encoder.setDepthStencilState(pipeline.depthStencilState)
        }
        self.encoder.setArgumentTable(RendererData.vertexTable, stages: .vertex)
        self.encoder.setArgumentTable(RendererData.fragmentTable, stages: .fragment)
    }

    func setMeshPipeline(pipeline: MeshPipeline) {
        self.encoder.setRenderPipelineState(pipeline.pipelineState)
        if pipeline.depthStencilState != nil {
            self.encoder.setDepthStencilState(pipeline.depthStencilState)
        }
        self.encoder.setArgumentTable(RendererData.objectTable, stages: .object)
        self.encoder.setArgumentTable(RendererData.meshTable, stages: .mesh)
        self.encoder.setArgumentTable(RendererData.fragmentTable, stages: .fragment)
    }

    func draw(primitiveType: MTLPrimitiveType, vertexCount: Int, vertexOffset: Int) {
        FrameAccumulator.current.directDrawCount += 1
        self.encoder.drawPrimitives(
            primitiveType: primitiveType, vertexStart: vertexOffset, vertexCount: vertexCount)
    }

    func drawIndexed(
        primitimeType: MTLPrimitiveType, buffer: Buffer, indexCount: Int, indexOffset: UInt64
    ) {
        FrameAccumulator.current.directDrawCount += 1
        let byteOffset = indexOffset * UInt64(MemoryLayout<UInt32>.size)
        self.encoder.drawIndexedPrimitives(
            primitiveType: primitimeType, indexCount: indexCount, indexType: .uint32,
            indexBuffer: buffer.getAddress() + byteOffset,
            indexBufferLength: indexCount * MemoryLayout<UInt32>.size)
    }

    func dispatchMesh(
        threadgroupsPerGrid: MTLSize, threadsPerObjectThreadgroup: MTLSize,
        threadsPerMeshThreadgroup: MTLSize
    ) {
        FrameAccumulator.current.directDrawCount += 1
        self.encoder.drawMeshThreadgroups(
            threadgroupsPerGrid: threadgroupsPerGrid,
            threadsPerObjectThreadgroup: threadsPerObjectThreadgroup,
            threadsPerMeshThreadgroup: threadsPerMeshThreadgroup)
    }

    func executeIndirect(icb: ICB, maxCommandCount: Int) {
        FrameAccumulator.current.executeIndirectCount += 1
        self.encoder.executeCommands(buffer: icb.cmdBuffer, range: 0..<maxCommandCount)
    }

    func setIFT(_ ift: MTLIntersectionFunctionTable, index: Int, stages: MTLRenderStages) {
        if stages.contains(.vertex) {
            RendererData.vertexTable.setResource(ift.gpuResourceID, bufferIndex: index)
        }
        if stages.contains(.fragment) {
            RendererData.fragmentTable.setResource(ift.gpuResourceID, bufferIndex: index)
        }
    }

    func setTexture(texture: Texture, index: Int, stages: MTLRenderStages) {
        if stages.contains(.vertex) {
            RendererData.vertexTable.setTexture(texture.texture.gpuResourceID, index: index)
        }
        if stages.contains(.fragment) {
            RendererData.fragmentTable.setTexture(texture.texture.gpuResourceID, index: index)
        }
        if stages.contains(.mesh) {
            RendererData.meshTable.setTexture(texture.texture.gpuResourceID, index: index)
        }
        if stages.contains(.object) {
            RendererData.objectTable.setTexture(texture.texture.gpuResourceID, index: index)
        }
    }

    func setTexture(texture: MTLTexture, index: Int, stages: MTLRenderStages) {
        if stages.contains(.vertex) {
            RendererData.vertexTable.setTexture(texture.gpuResourceID, index: index)
        }
        if stages.contains(.fragment) {
            RendererData.fragmentTable.setTexture(texture.gpuResourceID, index: index)
        }
        if stages.contains(.mesh) {
            RendererData.meshTable.setTexture(texture.gpuResourceID, index: index)
        }
        if stages.contains(.object) {
            RendererData.objectTable.setTexture(texture.gpuResourceID, index: index)
        }
    }

    func setBuffer(buf: Buffer, index: Int, stages: MTLRenderStages, offset: Int = 0) {
        if stages.contains(.vertex) {
            RendererData.vertexTable.setAddress(buf.getAddress() + UInt64(offset), index: index)
        }
        if stages.contains(.fragment) {
            RendererData.fragmentTable.setAddress(buf.getAddress() + UInt64(offset), index: index)
        }
        if stages.contains(.mesh) {
            RendererData.meshTable.setAddress(buf.getAddress() + UInt64(offset), index: index)
        }
        if stages.contains(.object) {
            RendererData.objectTable.setAddress(buf.getAddress() + UInt64(offset), index: index)
        }
    }

    func setBuffer(buf: MTLBuffer, index: Int, stages: MTLRenderStages, offset: Int = 0) {
        if stages.contains(.vertex) {
            RendererData.vertexTable.setAddress(buf.gpuAddress + UInt64(offset), index: index)
        }
        if stages.contains(.fragment) {
            RendererData.fragmentTable.setAddress(buf.gpuAddress + UInt64(offset), index: index)
        }
        if stages.contains(.mesh) {
            RendererData.meshTable.setAddress(buf.gpuAddress + UInt64(offset), index: index)
        }
        if stages.contains(.object) {
            RendererData.objectTable.setAddress(buf.gpuAddress + UInt64(offset), index: index)
        }
    }

    func setBytes(
        allocator: GPULinearAllocator, index: Int, bytes: UnsafeRawPointer, size: Int,
        stages: MTLRenderStages
    ) {
        let offset = allocator.allocate(size: size)
        allocator.writeData(data: bytes, offset: offset, size: size)
        if stages.contains(.vertex) {
            RendererData.vertexTable.setAddress(
                allocator.buffer.buffer.gpuAddress + UInt64(offset), index: index)
        }
        if stages.contains(.fragment) {
            RendererData.fragmentTable.setAddress(
                allocator.buffer.buffer.gpuAddress + UInt64(offset), index: index)
        }
        if stages.contains(.mesh) {
            RendererData.meshTable.setAddress(
                allocator.buffer.buffer.gpuAddress + UInt64(offset), index: index)
        }
        if stages.contains(.object) {
            RendererData.objectTable.setAddress(
                allocator.buffer.buffer.gpuAddress + UInt64(offset), index: index)
        }
    }

    func pushMarker(name: String) {
        self.encoder.pushDebugGroup(name)
    }

    func popMarker() {
        self.encoder.popDebugGroup()
    }

    func signalFenceAfterStage(stage: MTLStages) {
        self.encoder.updateFence(RendererData.gpuTimeline.fence, afterEncoderStages: stage)
    }

    func waitFenceBeforeStage(stage: MTLStages) {
        self.encoder.waitForFence(RendererData.gpuTimeline.fence, beforeEncoderStages: stage)
    }

    func intraPassBarrier(before: MTLStages, after: MTLStages) {
        self.encoder.barrier(
            afterEncoderStages: after, beforeEncoderStages: before, visibilityOptions: .device)
    }

    func consumerBarrier(before: MTLStages, after: MTLStages) {
        self.encoder.barrier(
            afterQueueStages: after, beforeStages: before, visibilityOptions: .device)
    }

    func producerBarrier(before: MTLStages, after: MTLStages) {
        self.encoder.barrier(
            afterStages: after, beforeQueueStages: before, visibilityOptions: .device)
    }
}
