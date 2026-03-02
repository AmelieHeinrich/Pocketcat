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

    func dispatch(threads: MTLSize, threadsPerGroup: MTLSize) {
        self.encoder.dispatchThreads(threadsPerGrid: threads, threadsPerThreadgroup: threadsPerGroup)
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
    
    func barrier(before: MTLStages, after: MTLStages) {
        self.encoder.barrier(afterEncoderStages: after, beforeEncoderStages: before, visibilityOptions: .device)
    }
}
