//
//  TLASBuildPass.swift
//  Pocketcat
//
//  Created by Amélie Heinrich on 10/03/2026.
//

import Metal

class TLASBuildPass: Pass {
    var cullPipe: ComputePipeline
    unowned var registry: SettingsRegistry

    init(settings: SettingsRegistry) {
        cullPipe = ComputePipeline(function: "cull_tlas")
        registry = settings
        
        settings.register(bool: "TLAS.BuildIndirect", label: "Indirect TLAS Build", default: false)
    }

    override func render(context: FrameContext) {
        guard let scene = context.scene else { return }
        
        let instanceCount = context.sceneBuffer.instanceCount
        if instanceCount == 0 {
            return
        }

        if registry.bool("TLAS.BuildIndirect") {
            gpuBuild(context: context, scene: scene)
        } else {
            cpuBuild(context: context, scene: scene)
        }
    }

    func cpuBuild(context: FrameContext, scene: RenderScene) {
        scene.tlas.resetInstanceBuffer()
        var instanceIdx: UInt32 = 0
        for entity in scene.entities {
            for (blasIdx, blas) in entity.mesh.blases.enumerated() {
                let materialIndex = Int(entity.mesh.instances[blasIdx].materialIndex)
                let alphaMode = entity.mesh.materials.count > 0 ? entity.mesh.materials[materialIndex].alphaMode : 0
                scene.tlas.addInstance(
                    blas: blas,
                    matrix: entity.transform,
                    opaque: alphaMode == 0,
                    userID: instanceIdx
                )
                instanceIdx += 1
            }
        }
        scene.tlas.update(frameIdx: context.frameIndex)

        let cp = context.cmdBuffer.beginComputePass(name: "Build TLAS (CPU)")
        cp.buildTLAS(tlas: scene.tlas)
        cp.end()
    }

    func gpuBuild(context: FrameContext, scene: RenderScene) {
        let instanceCount = context.sceneBuffer.instanceCount
        
        scene.tlas.updateFrameDescriptor(frameIdx: context.frameIndex)

        let cp = context.cmdBuffer.beginComputePass(name: "Build TLAS (GPU)")
        cp.pushMarker(name: "Reset Buffers")
        cp.resetBuffer(src: scene.tlas.instanceCountBuffers[context.frameIndex])
        cp.resetBuffer(src: scene.tlas.instanceBuffers[context.frameIndex])
        cp.popMarker()
        
        cp.pushMarker(name: "Cull Instances")
        cp.intraPassBarrier(before: .dispatch, after: .blit)
        cp.setPipeline(pipeline: cullPipe)
        cp.setBuffer(buf: context.sceneBuffer.buffer, index: 0)
        cp.setBuffer(buf: scene.tlas.instanceBuffers[context.frameIndex], index: 1)
        cp.setBuffer(buf: scene.tlas.instanceCountBuffers[context.frameIndex], index: 2)
        cp.dispatch(threads: MTLSizeMake((instanceCount + 63) / 64, 1, 1), threadsPerGroup: MTLSizeMake(64, 1, 1))
        cp.popMarker()

        cp.pushMarker(name: "Build TLAS")
        cp.intraPassBarrier(before: .accelerationStructure, after: .dispatch)
        cp.buildTLASIndirect(tlas: scene.tlas)
        cp.popMarker()
        cp.end()
    }
}
