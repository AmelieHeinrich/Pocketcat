//
//  TLASBuildPass.swift
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 10/03/2026.
//

import Metal

class TLASBuildPass : Pass {
    var cullPipe: ComputePipeline
    
    override init() {
        cullPipe = ComputePipeline(function: "cull_tlas")
    }
    
    override func render(context: FrameContext) {
        if let scene = context.scene {
            let instanceCount = context.sceneBuffer.instanceCount
            if instanceCount == 0 {
                return
            }
            
            let cp = context.cmdBuffer.beginComputePass(name: "Build TLAS")
            cp.resetBuffer(src: scene.tlas.instanceCountBuffer)
            
            cp.intraPassBarrier(before: .dispatch, after: .blit)
            cp.setPipeline(pipeline: cullPipe)
            cp.setBuffer(buf: context.sceneBuffer.buffer, index: 0)
            cp.setBuffer(buf: scene.tlas.instanceBuffer, index: 1)
            cp.setBuffer(buf: scene.tlas.instanceCountBuffer, index: 2)
            cp.dispatch(threads: MTLSizeMake((instanceCount + 63) / 64, 1, 1), threadsPerGroup: MTLSizeMake(64, 1, 1))
            
            cp.intraPassBarrier(before: .accelerationStructure, after: .dispatch)
            cp.buildTLASIndirect(tlas: scene.tlas)
            cp.end()
        }
    }
}
