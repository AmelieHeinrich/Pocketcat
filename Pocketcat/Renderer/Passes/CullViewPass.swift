//
//  CullViewPass.swift
//  Pocketcat
//
//  Created by Amélie Heinrich on 20/03/2026.
//

import Metal
import simd

class CullViewPass: Pass {
    private let icbResetPipe: ComputePipeline
    private let vertexPipe: ComputePipeline
    private let meshPipe: ComputePipeline
    private var vertexICB: ICB
    private var meshICB: ICB
    private let instanceIDBuffer: Buffer
    private unowned let settings: RendererSettings
    
    init(settings: RendererSettings) {
        self.settings = settings
        
        self.icbResetPipe = ComputePipeline(function: "reset_icb", name: "Reset ICB")
        self.vertexPipe = ComputePipeline(function: "vertex_geometry_cull", name: "Cull Instances (VS)")
        self.meshPipe = ComputePipeline(function: "mesh_geometry_cull", name: "Cull Instances (MS)")
        
        self.vertexICB = ICB(inherit: false, commandTypes: .drawIndexed, maxCommandCount: 65536)
        self.vertexICB.setName(label: "Vertex Forward ICB")
        
        self.meshICB = ICB(inherit: false, commandTypes: .drawMeshThreadgroups, maxCommandCount: 65536)
        self.meshICB.setName(label: "Mesh Forward ICB")
        
        self.instanceIDBuffer = Buffer(size: 65536 * MemoryLayout<UInt32>.size)
        instanceIDBuffer.setName(name: "Instance ID Buffer (Mesh ICB)")
    }
    
    override func render(context: FrameContext) {
        guard (context.scene != nil) else { return }
        
        if settings.useMeshShader {
            meshCull(context: context)
        } else {
            vertexCull(context: context)
        }
    }
    
    func vertexCull(context: FrameContext) {
        var maxInstanceCount = 65536
        var instanceCount = context.sceneBuffer.instanceCount
        let resetTgCountX = (Int(maxInstanceCount) + 63) / 64
        let cullTgCountX = (Int(instanceCount) + 63) / 64
        
        let cp = context.cmdBuffer.beginComputePass(name: "Reset & Cull Instances")
        cp.pushMarker(name: "Reset")
        cp.setPipeline(pipeline: icbResetPipe)
        cp.setBuffer(buf: vertexICB.buffer, index: 0)
        cp.setBytes(allocator: context.allocator, index: 1, bytes: &maxInstanceCount, size: MemoryLayout<Int>.size)
        cp.dispatch(threads: MTLSizeMake(resetTgCountX, 1, 1), threadsPerGroup: MTLSizeMake(64, 1, 1))
        cp.popMarker()
        cp.pushMarker(name: "Cull")
        
        cp.intraPassBarrier(before: .dispatch, after: .dispatch)
        cp.setPipeline(pipeline: vertexPipe)
        cp.setBuffer(buf: context.sceneBuffer.buffer, index: 0)
        cp.setBuffer(buf: vertexICB.buffer, index: 1)
        cp.setBytes(allocator: context.allocator, index: 2, bytes: &instanceCount, size: MemoryLayout<UInt32>.size)
        cp.dispatch(threads: MTLSizeMake(cullTgCountX, 1, 1), threadsPerGroup: MTLSizeMake(64, 1, 1))
        cp.popMarker()
        
        cp.end()
        
        context.resources.register(vertexICB, for: "MainViewICB")
    }
    
    func meshCull(context: FrameContext) {
        var maxInstanceCount = 65536
        var instanceCount = context.sceneBuffer.instanceCount
        let resetTgCountX = (Int(maxInstanceCount) + 63) / 64
        let cullTgCountX = (Int(instanceCount) + 63) / 64
        
        let cp = context.cmdBuffer.beginComputePass(name: "Reset & Cull Instances")
        cp.pushMarker(name: "Reset")
        cp.setPipeline(pipeline: icbResetPipe)
        cp.setBuffer(buf: meshICB.buffer, index: 0)
        cp.setBytes(allocator: context.allocator, index: 1, bytes: &maxInstanceCount, size: MemoryLayout<Int>.size)
        cp.dispatch(threads: MTLSizeMake(resetTgCountX, 1, 1), threadsPerGroup: MTLSizeMake(64, 1, 1))
        cp.popMarker()
        
        cp.pushMarker(name: "Cull")
        cp.intraPassBarrier(before: .dispatch, after: .dispatch)
        cp.setPipeline(pipeline: meshPipe)
        cp.setBuffer(buf: context.sceneBuffer.buffer, index: 0)
        cp.setBuffer(buf: meshICB.buffer, index: 1)
        cp.setBytes(allocator: context.allocator, index: 2, bytes: &instanceCount, size: MemoryLayout<UInt32>.size)
        cp.setBuffer(buf: instanceIDBuffer, index: 3)
        cp.dispatch(threads: MTLSizeMake(cullTgCountX, 1, 1), threadsPerGroup: MTLSizeMake(64, 1, 1))
        cp.popMarker()
        
        cp.end()
        
        context.resources.register(meshICB, for: "MainViewICB")
    }
}
