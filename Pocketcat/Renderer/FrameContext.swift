//
//  FrameContext.swift
//  Pocketcat
//
//  Created by Amélie Heinrich on 02/03/2026.
//

internal import QuartzCore
import simd

struct CameraData {
    var view: simd_float4x4
    var projection: simd_float4x4
    var viewProjection: simd_float4x4
    var viewProjectionNoJitter: simd_float4x4
    var inverseView: simd_float4x4
    var inverseProjection: simd_float4x4
    var inverseViewProjection: simd_float4x4
    var position: SIMD3<Float>
    var direction: SIMD3<Float>
    var near: Float
    var far: Float

    init() {
        view = .identity
        projection = .identity
        viewProjection = .identity
        viewProjectionNoJitter = .identity
        inverseView = .identity
        inverseProjection = .identity
        inverseViewProjection = .identity
        position = .zero
        direction = SIMD3<Float>(0, 0, -1)
        near = 0.1
        far = 1000.0
    }
}

struct FrameContext {
    var camera: CameraData
    var cmdBuffer: CommandBuffer
    var drawable: CAMetalDrawable
    var resources: ResourceRegistry
    var scene: RenderScene?
    var sceneBuffer: SceneBufferBuilder
    var frameIndex: Int
    var allocator: GPULinearAllocator
}

extension Camera {
    func makeCameraData() -> CameraData {
        var data = CameraData()
        data.view = viewMatrix
        data.projection = projectionMatrix
        data.viewProjection = viewProjection
        data.viewProjectionNoJitter = viewProjectionNoJitter
        data.inverseView = viewMatrix.inverse
        data.inverseProjection = projectionMatrix.inverse
        data.inverseViewProjection = viewProjection.inverse
        data.position = position
        data.direction = forward
        data.near = near
        data.far = far
        return data
    }
}
