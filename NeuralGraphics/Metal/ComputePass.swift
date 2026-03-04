//
//  ComputePass.swift
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 02/03/2026.
//

import Metal

class ComputePass {
    var encoder: MTL4ComputeCommandEncoder

    init(label: String, cmdBuffer: MTL4CommandBuffer) {
        self.encoder = cmdBuffer.makeComputeCommandEncoder()!
        self.encoder.label = label
    }

    func end() {
        self.encoder.endEncoding()
    }

    func setPipeline(pipeline: ComputePipeline) {
        self.encoder.setComputePipelineState(pipeline.pipelineState)
        self.encoder.setArgumentTable(RendererData.computeTable)
    }

    func setBuffer(buf: Buffer, index: Int, offset: Int = 0) {
        RendererData.computeTable.setAddress(buf.getAddress() + UInt64(offset), index: index)
    }

    func setTexture(texture: Texture, index: Int) {
        RendererData.computeTable.setTexture(texture.texture.gpuResourceID, index: index)
    }
    
    func setTexture(texture: MTLTexture, index: Int) {
        RendererData.computeTable.setTexture(texture.gpuResourceID, index: index)
    }

    func setBytes(allocator: GPULinearAllocator, index: Int, bytes: UnsafeRawPointer, size: Int) {
        let offset = allocator.allocate(size: size)
        allocator.writeData(data: bytes, offset: offset, size: size)
        RendererData.computeTable.setAddress(allocator.buffer.buffer.gpuAddress + UInt64(offset), index: index)
    }
    
    func dispatch(threads: MTLSize, threadsPerGroup: MTLSize) {
        self.encoder.dispatchThreadgroups(threadgroupsPerGrid: threads, threadsPerThreadgroup: threadsPerGroup)
    }
    
    func buildBLAS(blas: BLAS) {
        self.encoder.build(destinationAccelerationStructure: blas.accelerationStructure, descriptor: blas.descriptor, scratchBuffer: MTL4BufferRangeMake(blas.scratchBuffer.getAddress(), UInt64(blas.scratchBuffer.size)))
    }

    func buildTLAS(tlas: TLAS) {
        self.encoder.build(destinationAccelerationStructure: tlas.tlas, descriptor: tlas.descriptor, scratchBuffer: MTL4BufferRangeMake(tlas.scratchBuffer.getAddress(), UInt64(tlas.scratchBuffer.size)))
    }
    
    func copyTexture(src: Texture, dst: Texture) {
        self.encoder.copy(sourceTexture: src.texture, destinationTexture: dst.texture)
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
        self.encoder.barrier(afterEncoderStages: after, beforeEncoderStages: before, visibilityOptions: .device)
    }
    
    func consumerBarrier(before: MTLStages, after: MTLStages) {
        self.encoder.barrier(afterQueueStages: after, beforeStages: before, visibilityOptions: .device)
    }
    
    func producerBarrier(before: MTLStages, after: MTLStages) {
        self.encoder.barrier(afterStages: after, beforeQueueStages: before, visibilityOptions: .device)
    }
}
