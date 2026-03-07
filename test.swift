//
//  Renderer.swift
//  Testbed
//
//  Created by Amélie Heinrich on 07/03/2026.
//

// Our platform independent renderer class

import Metal
import MetalKit
import simd

// The 256 byte aligned size of our uniform structure
let alignedUniformsSize = (MemoryLayout<Uniforms>.size + 0xFF) & -0x100

let maxBuffersInFlight = 3

nonisolated enum RendererError: Error {
    case badVertexDescriptor
}

class Renderer: NSObject, MTKViewDelegate {

    public let device: MTLDevice

    let commandQueue: MTL4CommandQueue
    let commandBuffer: MTL4CommandBuffer
    let commandAllocators: [MTL4CommandAllocator]
    let commandQueueResidencySet: MTLResidencySet
    let vertexArgumentTable: MTL4ArgumentTable
    let fragmentArgumentTable: MTL4ArgumentTable

    let endFrameEvent: MTLSharedEvent
    var frameIndex = 0

    var dynamicUniformBuffer: MTLBuffer
    var pipelineState: MTLRenderPipelineState
    var depthState: MTLDepthStencilState
    var colorMap: MTLTexture

    var uniformBufferOffset = 0

    var uniformBufferIndex = 0

    var uniforms: UnsafeMutablePointer<Uniforms>

    var projectionMatrix: matrix_float4x4 = matrix_float4x4()

    var rotation: Float = 0

    var mesh: MTKMesh

    @MainActor
    init?(metalKitView: MTKView) {
        let device = metalKitView.device!
        self.device = device

        self.commandQueue = device.makeMTL4CommandQueue()!
        self.commandBuffer = device.makeCommandBuffer()!
        self.commandAllocators = (0...maxBuffersInFlight).map { _ in device.makeCommandAllocator()!
        }

        let argTableDesc = MTL4ArgumentTableDescriptor()
        argTableDesc.maxBufferBindCount = 4
        self.vertexArgumentTable = try! device.makeArgumentTable(descriptor: argTableDesc)
        argTableDesc.maxTextureBindCount = 1
        self.fragmentArgumentTable = try! device.makeArgumentTable(descriptor: argTableDesc)

        self.endFrameEvent = device.makeSharedEvent()!
        frameIndex = maxBuffersInFlight
        self.endFrameEvent.signaledValue = UInt64(frameIndex - 1)

        let uniformBufferSize = alignedUniformsSize * maxBuffersInFlight

        guard
            let buffer = self.device.makeBuffer(
                length: uniformBufferSize, options: [MTLResourceOptions.storageModeShared])
        else { return nil }
        dynamicUniformBuffer = buffer

        self.dynamicUniformBuffer.label = "UniformBuffer"

        uniforms = UnsafeMutableRawPointer(dynamicUniformBuffer.contents()).bindMemory(
            to: Uniforms.self, capacity: 1)

        metalKitView.depthStencilPixelFormat = MTLPixelFormat.depth32Float_stencil8
        metalKitView.colorPixelFormat = MTLPixelFormat.bgra8Unorm_srgb
        metalKitView.sampleCount = 1

        let mtlVertexDescriptor = Renderer.buildMetalVertexDescriptor()

        do {
            pipelineState = try Renderer.buildRenderPipelineWithDevice(
                device: device,
                metalKitView: metalKitView,
                mtlVertexDescriptor: mtlVertexDescriptor)
        } catch {
            print("Unable to compile render pipeline state.  Error info: \(error)")
            return nil
        }

        let depthStateDescriptor = MTLDepthStencilDescriptor()
        depthStateDescriptor.depthCompareFunction = MTLCompareFunction.less
        depthStateDescriptor.isDepthWriteEnabled = true
        guard let state = device.makeDepthStencilState(descriptor: depthStateDescriptor) else {
            return nil
        }
        depthState = state

        do {
            mesh = try Renderer.buildMesh(device: device, mtlVertexDescriptor: mtlVertexDescriptor)
        } catch {
            print("Unable to build MetalKit Mesh. Error info: \(error)")
            return nil
        }

        do {
            colorMap = try Renderer.loadTexture(device: device, textureName: "ColorMap")
        } catch {
            print("Unable to load texture. Error info: \(error)")
            return nil
        }

        let residencySetDesc = MTLResidencySetDescriptor()
        residencySetDesc.initialCapacity = mesh.vertexBuffers.count + mesh.submeshes.count + 2  // color map + uniforms buffer
        let residencySet = try! self.device.makeResidencySet(descriptor: residencySetDesc)
        residencySet.addAllocations(mesh.vertexBuffers.map { $0.buffer })
        residencySet.addAllocations(mesh.submeshes.map { $0.indexBuffer.buffer })
        residencySet.addAllocations([colorMap, dynamicUniformBuffer])
        residencySet.commit()
        commandQueue.addResidencySet(residencySet)
        commandQueueResidencySet = residencySet

        super.init()

    }

    class func buildMetalVertexDescriptor() -> MTLVertexDescriptor {
        // Create a Metal vertex descriptor specifying how vertices will by laid out for input into our render
        //   pipeline and how we'll layout our Model IO vertices

        let mtlVertexDescriptor = MTLVertexDescriptor()

        mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].format =
            MTLVertexFormat.float3
        mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].offset = 0
        mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].bufferIndex =
            BufferIndex.meshPositions.rawValue

        mtlVertexDescriptor.attributes[VertexAttribute.texcoord.rawValue].format =
            MTLVertexFormat.float2
        mtlVertexDescriptor.attributes[VertexAttribute.texcoord.rawValue].offset = 0
        mtlVertexDescriptor.attributes[VertexAttribute.texcoord.rawValue].bufferIndex =
            BufferIndex.meshGenerics.rawValue

        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stride = 12
        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepRate = 1
        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepFunction =
            MTLVertexStepFunction.perVertex

        mtlVertexDescriptor.layouts[BufferIndex.meshGenerics.rawValue].stride = 8
        mtlVertexDescriptor.layouts[BufferIndex.meshGenerics.rawValue].stepRate = 1
        mtlVertexDescriptor.layouts[BufferIndex.meshGenerics.rawValue].stepFunction =
            MTLVertexStepFunction.perVertex

        return mtlVertexDescriptor
    }

    @MainActor
    class func buildRenderPipelineWithDevice(
        device: MTLDevice,
        metalKitView: MTKView,
        mtlVertexDescriptor: MTLVertexDescriptor
    ) throws -> MTLRenderPipelineState {
        /// Build a render state pipeline object

        let library = device.makeDefaultLibrary()
        let compiler = try device.makeCompiler(descriptor: MTL4CompilerDescriptor())

        let vertexFunctionDescriptor = MTL4LibraryFunctionDescriptor()
        vertexFunctionDescriptor.library = library
        vertexFunctionDescriptor.name = "vertexShader"
        let fragmentFunctionDescriptor = MTL4LibraryFunctionDescriptor()
        fragmentFunctionDescriptor.library = library
        fragmentFunctionDescriptor.name = "fragmentShader"

        let pipelineDescriptor = MTL4RenderPipelineDescriptor()
        pipelineDescriptor.label = "RenderPipeline"
        pipelineDescriptor.rasterSampleCount = metalKitView.sampleCount
        pipelineDescriptor.vertexFunctionDescriptor = vertexFunctionDescriptor
        pipelineDescriptor.fragmentFunctionDescriptor = fragmentFunctionDescriptor
        pipelineDescriptor.vertexDescriptor = mtlVertexDescriptor

        pipelineDescriptor.colorAttachments[0].pixelFormat = metalKitView.colorPixelFormat

        return try compiler.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }

    class func buildMesh(
        device: MTLDevice,
        mtlVertexDescriptor: MTLVertexDescriptor
    ) throws -> MTKMesh {
        /// Create and condition mesh data to feed into a pipeline using the given vertex descriptor

        let metalAllocator = MTKMeshBufferAllocator(device: device)

        let mdlMesh = MDLMesh.newBox(
            withDimensions: SIMD3<Float>(4, 4, 4),
            segments: SIMD3<UInt32>(2, 2, 2),
            geometryType: MDLGeometryType.triangles,
            inwardNormals: false,
            allocator: metalAllocator)

        let mdlVertexDescriptor = MTKModelIOVertexDescriptorFromMetal(mtlVertexDescriptor)

        guard let attributes = mdlVertexDescriptor.attributes as? [MDLVertexAttribute] else {
            throw RendererError.badVertexDescriptor
        }
        attributes[VertexAttribute.position.rawValue].name = MDLVertexAttributePosition
        attributes[VertexAttribute.texcoord.rawValue].name = MDLVertexAttributeTextureCoordinate

        mdlMesh.vertexDescriptor = mdlVertexDescriptor

        return try MTKMesh(mesh: mdlMesh, device: device)
    }

    class func loadTexture(
        device: MTLDevice,
        textureName: String
    ) throws -> MTLTexture {
        /// Load texture data with optimal parameters for sampling

        let textureLoader = MTKTextureLoader(device: device)

        let textureLoaderOptions = [
            MTKTextureLoader.Option.textureUsage: NSNumber(
                value: MTLTextureUsage.shaderRead.rawValue),
            MTKTextureLoader.Option.textureStorageMode: NSNumber(
                value: MTLStorageMode.`private`.rawValue),
        ]

        return try textureLoader.newTexture(
            name: textureName,
            scaleFactor: 1.0,
            bundle: nil,
            options: textureLoaderOptions)

    }

    private func updateDynamicBufferState() {
        /// Update the state of our uniform buffers before rendering

        uniformBufferIndex = (uniformBufferIndex + 1) % maxBuffersInFlight

        uniformBufferOffset = alignedUniformsSize * uniformBufferIndex

        uniforms = UnsafeMutableRawPointer(dynamicUniformBuffer.contents() + uniformBufferOffset)
            .bindMemory(to: Uniforms.self, capacity: 1)
    }

    private func updateGameState() {
        /// Update any game state before rendering

        uniforms[0].projectionMatrix = projectionMatrix

        let rotationAxis = SIMD3<Float>(1, 1, 0)
        let modelMatrix = matrix4x4_rotation(radians: rotation, axis: rotationAxis)
        let viewMatrix = matrix4x4_translation(0.0, 0.0, -8.0)
        uniforms[0].modelViewMatrix = simd_mul(viewMatrix, modelMatrix)
        rotation += 0.01
    }

    func draw(in view: MTKView) {
        /// Per frame updates hare

        guard let drawable = view.currentDrawable else { return }

        /// Delay getting the currentRenderPassDescriptor until we absolutely need it to avoid
        ///   holding onto the drawable and blocking the display pipeline any longer than necessary
        guard let renderPassDescriptor = view.currentMTL4RenderPassDescriptor else { return }

        let previousValueToWaitFor = self.frameIndex - maxBuffersInFlight
        self.endFrameEvent.wait(untilSignaledValue: UInt64(previousValueToWaitFor), timeoutMS: 10)
        let commandAllocator = self.commandAllocators[uniformBufferIndex]
        commandAllocator.reset()
        commandBuffer.beginCommandBuffer(allocator: commandAllocator)

        self.updateDynamicBufferState()

        self.updateGameState()

        guard
            let renderEncoder = self.commandBuffer.makeRenderCommandEncoder(
                descriptor: renderPassDescriptor)
        else {
            fatalError("Failed to create render command encoder")
        }

        /// Final pass rendering code here
        renderEncoder.label = "Primary Render Encoder"

        renderEncoder.pushDebugGroup("Draw Box")

        renderEncoder.setCullMode(.back)

        renderEncoder.setFrontFacing(.counterClockwise)

        renderEncoder.setRenderPipelineState(pipelineState)

        renderEncoder.setDepthStencilState(depthState)

        renderEncoder.setArgumentTable(self.vertexArgumentTable, stages: .vertex)
        renderEncoder.setArgumentTable(self.fragmentArgumentTable, stages: .fragment)

        self.vertexArgumentTable.setAddress(
            dynamicUniformBuffer.gpuAddress + UInt64(uniformBufferOffset),
            index: BufferIndex.uniforms.rawValue)
        self.fragmentArgumentTable.setAddress(
            dynamicUniformBuffer.gpuAddress + UInt64(uniformBufferOffset),
            index: BufferIndex.uniforms.rawValue)

        for (index, element) in mesh.vertexDescriptor.layouts.enumerated() {
            guard let layout = element as? MDLVertexBufferLayout else {
                return
            }

            if layout.stride != 0 {
                let buffer = mesh.vertexBuffers[index]
                self.vertexArgumentTable.setAddress(
                    buffer.buffer.gpuAddress + UInt64(buffer.offset), index: index)
            }
        }

        self.fragmentArgumentTable.setTexture(
            colorMap.gpuResourceID, index: TextureIndex.color.rawValue)

        for submesh in mesh.submeshes {
            renderEncoder.drawIndexedPrimitives(
                primitiveType: submesh.primitiveType,
                indexCount: submesh.indexCount,
                indexType: submesh.indexType,
                indexBuffer: submesh.indexBuffer.buffer.gpuAddress
                    + UInt64(submesh.indexBuffer.offset),
                indexBufferLength: submesh.indexBuffer.buffer.length)
        }

        renderEncoder.popDebugGroup()

        renderEncoder.endEncoding()

        commandBuffer.useResidencySet((view.layer as! CAMetalLayer).residencySet)
        commandBuffer.endCommandBuffer()

        commandQueue.waitForDrawable(drawable)
        commandQueue.commit([commandBuffer])
        commandQueue.signalDrawable(drawable)
        commandQueue.signalEvent(self.endFrameEvent, value: UInt64(self.frameIndex))
        self.frameIndex += 1
        drawable.present()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        /// Respond to drawable size or orientation changes here

        let aspect = Float(size.width) / Float(size.height)
        projectionMatrix = matrix_perspective_right_hand(
            fovyRadians: radians_from_degrees(65), aspectRatio: aspect, nearZ: 0.1, farZ: 100.0)
    }
}

// Generic matrix math utility functions
func matrix4x4_rotation(radians: Float, axis: SIMD3<Float>) -> matrix_float4x4 {
    let unitAxis = normalize(axis)
    let ct = cosf(radians)
    let st = sinf(radians)
    let ci = 1 - ct
    let x = unitAxis.x
    let y = unitAxis.y
    let z = unitAxis.z
    return matrix_float4x4.init(
        columns: (
            vector_float4(ct + x * x * ci, y * x * ci + z * st, z * x * ci - y * st, 0),
            vector_float4(x * y * ci - z * st, ct + y * y * ci, z * y * ci + x * st, 0),
            vector_float4(x * z * ci + y * st, y * z * ci - x * st, ct + z * z * ci, 0),
            vector_float4(0, 0, 0, 1)
        ))
}

func matrix4x4_translation(_ translationX: Float, _ translationY: Float, _ translationZ: Float)
    -> matrix_float4x4
{
    return matrix_float4x4.init(
        columns: (
            vector_float4(1, 0, 0, 0),
            vector_float4(0, 1, 0, 0),
            vector_float4(0, 0, 1, 0),
            vector_float4(translationX, translationY, translationZ, 1)
        ))
}

func matrix_perspective_right_hand(
    fovyRadians fovy: Float, aspectRatio: Float, nearZ: Float, farZ: Float
) -> matrix_float4x4 {
    let ys = 1 / tanf(fovy * 0.5)
    let xs = ys / aspectRatio
    let zs = farZ / (nearZ - farZ)
    return matrix_float4x4.init(
        columns: (
            vector_float4(xs, 0, 0, 0),
            vector_float4(0, ys, 0, 0),
            vector_float4(0, 0, zs, -1),
            vector_float4(0, 0, zs * nearZ, 0)
        ))
}

func radians_from_degrees(_ degrees: Float) -> Float {
    return (degrees / 180) * .pi
}
