//
//  VisibilityBuffer.swift
//  Pocketcat
//
//  Created by Amélie Heinrich on 20/03/2026.
//

import Metal
import simd

class VisibilityBufferPass: Pass {
    private let vertexPipe: RenderPipeline
    private let meshPipe: MeshPipeline
    private var visibilityTexture: Texture
    private var previousVisibilityTexture: Texture
    private var motionVectorTexture: Texture
    private var previousMotionVectorTexture: Texture
    private var depthTexture: Texture
    private var previousDepth: Texture
    private unowned let registry: SettingsRegistry

    init(registry: SettingsRegistry) {
        var pipelineDesc = RenderPipelineDescriptor()
        pipelineDesc.name = "Visibility (VS)"
        pipelineDesc.vertexFunction = "visibility_vs"
        pipelineDesc.fragmentFunction = "visibility_fs_vs"
        pipelineDesc.pixelFormats = [.rg32Uint, .rg16Float]
        pipelineDesc.depthEnabled = true
        pipelineDesc.depthFormat = .depth32Float
        pipelineDesc.depthCompareOp = .less
        pipelineDesc.depthWriteEnabled = true
        pipelineDesc.supportsIndirect = true

        var meshPipelineDesc = MeshPipelineDescriptor()
        meshPipelineDesc.name = "Visibility (MS)"
        meshPipelineDesc.objectFunction = "visibility_os"
        meshPipelineDesc.meshFunction = "visibility_ms"
        meshPipelineDesc.fragmentFunction = "visibility_fs_ms"
        meshPipelineDesc.pixelFormats = [.rg32Uint, .rg16Float]
        meshPipelineDesc.depthEnabled = true
        meshPipelineDesc.depthFormat = .depth32Float
        meshPipelineDesc.depthCompareOp = .less
        meshPipelineDesc.depthWriteEnabled = true
        meshPipelineDesc.supportsIndirect = true

        self.vertexPipe = RenderPipeline(descriptor: pipelineDesc)
        self.meshPipe = MeshPipeline(descriptor: meshPipelineDesc)

        let visDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg32Uint, width: 1, height: 1, mipmapped: false)
        visDesc.usage = [.shaderRead, .renderTarget, .shaderWrite]

        let motionDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 1, height: 1, mipmapped: false)
        motionDesc.usage = [.shaderRead, .renderTarget, .shaderWrite]

        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float, width: 1, height: 1, mipmapped: false)
        depthDesc.usage = [.shaderRead, .renderTarget, .shaderWrite]

        self.visibilityTexture = Texture(descriptor: visDesc)
        self.visibilityTexture.setLabel(name: "Visibility Texture")
        
        self.previousVisibilityTexture = Texture(descriptor: visDesc)
        self.previousVisibilityTexture.setLabel(name: "PREVIOUS Visibility Texture")

        self.motionVectorTexture = Texture(descriptor: motionDesc)
        self.motionVectorTexture.setLabel(name: "Motion Vectors")

        self.previousMotionVectorTexture = Texture(descriptor: motionDesc)
        self.previousMotionVectorTexture.setLabel(name: "PREVIOUS Motion Vectors")

        self.depthTexture = Texture(descriptor: depthDesc)
        self.depthTexture.setLabel(name: "Visibility Depth")
        
        self.previousDepth = Texture(descriptor: depthDesc)
        self.previousDepth.setLabel(name: "PREVIOUS Visibility Depth")

        self.registry = registry
        registry.register(bool: "Visibility.MeshShader", label: "Mesh Shaders", default: true)
        registry.register(int: "Visibility.ForcedLOD", label: "Forced LOD", default: 0, range: 0...4)

        super.init()
    }

    override func resize(renderWidth: Int, renderHeight: Int, outputWidth: Int, outputHeight: Int) {
        visibilityTexture.resize(width: renderWidth, height: renderHeight)
        motionVectorTexture.resize(width: renderWidth, height: renderHeight)
        previousMotionVectorTexture.resize(width: renderWidth, height: renderHeight)
        depthTexture.resize(width: renderWidth, height: renderHeight)
        previousDepth.resize(width: renderWidth, height: renderHeight)
        previousVisibilityTexture.resize(width: renderWidth, height: renderHeight)
    }

    override func render(context: FrameContext) {
        let icb = context.resources.get("MainViewICB") as ICB?
        guard let icb = icb else { return }

        var rpDesc = RenderPassDescriptor()
        rpDesc.setName(name: "Visibility Buffer")
        rpDesc.addAttachment(texture: self.visibilityTexture)
        rpDesc.addAttachment(texture: self.motionVectorTexture)
        rpDesc.setDepthAttachment(texture: self.depthTexture)

        let rp = context.cmdBuffer.beginRenderPass(descriptor: rpDesc)
        rp.consumerBarrier(before: [.vertex, .object], after: [.dispatch])
        if registry.bool("Visibility.MeshShader") {
            rp.setMeshPipeline(pipeline: meshPipe)
        } else {
            rp.setPipeline(pipeline: vertexPipe)
        }
        rp.executeIndirect(icb: icb, maxCommandCount: 65536)
        rp.end()
    
        context.resources.register(visibilityTexture, for: "Visibility")
        context.resources.register(previousVisibilityTexture, for: "History.Visibility")
        context.resources.register(previousDepth, for: "History.GBuffer.Depth")
        context.resources.register(motionVectorTexture, for: "GBuffer.MotionVectors")
        context.resources.register(previousMotionVectorTexture, for: "History.GBuffer.MotionVectors")
        context.resources.register(depthTexture, for: "GBuffer.Depth")

        context.resources.addVisualizer(texture: visibilityTexture, label: "Visibility.InstanceID",
            fragmentFunction: "texviz_visibility_instance_id_fs")
        context.resources.addVisualizer(texture: visibilityTexture, label: "Visibility.MeshletID",
            fragmentFunction: "texviz_visibility_meshlet_id_fs")
        context.resources.addVisualizer(texture: visibilityTexture, label: "Visibility.PrimitiveID",
            fragmentFunction: "texviz_visibility_primitive_id_fs")
        context.resources.addVisualizer(texture: depthTexture, label: "GBuffer.Depth",
            fragmentFunction: "texviz_depth_fs")
        context.resources.addVisualizer(texture: motionVectorTexture, label: "GBuffer.MotionVectors",
            fragmentFunction: "texviz_motion_vectors_fs")
    }
    
    override func postRender(encoder: ComputePass) {
        encoder.copyTexture(src: self.visibilityTexture, dst: self.previousVisibilityTexture)
        encoder.copyTexture(src: self.depthTexture, dst: self.previousDepth)
        encoder.copyTexture(src: self.motionVectorTexture, dst: self.previousMotionVectorTexture)
    }
}
