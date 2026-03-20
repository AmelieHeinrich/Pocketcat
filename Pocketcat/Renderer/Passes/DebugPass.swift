//
//  DebugPass.swift
//  Pocketcat
//
//  Created by Amélie Heinrich on 03/03/2026.
//

import Metal
internal import QuartzCore
import simd

// Matches DebugVertex in Debug.metal.
// packed_float3 = 12 bytes, packed_float4 = 16 bytes → 28 bytes total.
private struct DebugVertex {
    var x: Float, y: Float, z: Float  // packed_float3 Position
    var r: Float, g: Float, b: Float, a: Float  // packed_float4 Color
}

// Matches DebugData in Debug.metal
private struct DebugData {
    var camera: simd_float4x4
}

class DebugPass: Pass {
    static let shared = DebugPass()

    weak var settings: RendererSettings? = nil

    private let pipelineNoDepth: RenderPipeline
    private let pipelineDepth: RenderPipeline
    private let icbPipe: ComputePipeline
    private let icbs: [ICB]
    private let vertexBuffers: [Buffer]
    private let maxVertices: Int = 65536 * 4
    private var vertices: [DebugVertex] = []

    private override init() {
        self.icbPipe = ComputePipeline(function: "debug_generate_icb", name: "Generate Debug ICB")
        self.icbs = (0..<3).map { i in
            let icb = ICB(inherit: true, commandTypes: .draw, maxCommandCount: 1)
            icb.setName(label: "Debug ICB \(i)")
            return icb
        }

        var descNoDepth = RenderPipelineDescriptor()
        descNoDepth.name = "Debug Line Pipeline"
        descNoDepth.vertexFunction = "debug_vs"
        descNoDepth.fragmentFunction = "debug_fs"
        descNoDepth.pixelFormats.append(.bgra8Unorm)
        descNoDepth.supportsIndirect = true
        self.pipelineNoDepth = RenderPipeline(descriptor: descNoDepth)

        var descDepth = RenderPipelineDescriptor()
        descDepth.name = "Debug Line Pipeline (Depth)"
        descDepth.vertexFunction = "debug_vs"
        descDepth.fragmentFunction = "debug_fs"
        descDepth.pixelFormats.append(.bgra8Unorm)
        descDepth.depthFormat = .depth32Float
        descDepth.depthEnabled = true
        descDepth.depthWriteEnabled = false  // read-only: don't corrupt the forward depth buffer
        descDepth.depthCompareOp = .lessEqual
        descDepth.supportsIndirect = true
        self.pipelineDepth = RenderPipeline(descriptor: descDepth)

        let stride = MemoryLayout<DebugVertex>.stride
        self.vertices.reserveCapacity(maxVertices)
        self.vertexBuffers = (0..<3).map { i in
            let buf = Buffer(size: stride * (65536 * 2))
            buf.setName(name: "Debug Vertex Buffer \(i)")
            return buf
        }

        super.init()
    }

    // MARK: - Primitives

    func drawLine(from: SIMD3<Float>, to: SIMD3<Float>, color: SIMD4<Float> = SIMD4(1, 1, 1, 1)) {
        guard vertices.count + 2 <= maxVertices else { return }
        vertices.append(
            DebugVertex(
                x: from.x, y: from.y, z: from.z,
                r: color.x, g: color.y, b: color.z, a: color.w))
        vertices.append(
            DebugVertex(
                x: to.x, y: to.y, z: to.z,
                r: color.x, g: color.y, b: color.z, a: color.w))
    }

    // MARK: - Box / Cube

    /// Wireframe axis-aligned box from `min` to `max`.
    func drawBox(min: SIMD3<Float>, max: SIMD3<Float>, color: SIMD4<Float> = SIMD4(1, 1, 1, 1)) {
        let c: [SIMD3<Float>] = [
            SIMD3(min.x, min.y, min.z), SIMD3(max.x, min.y, min.z),
            SIMD3(max.x, min.y, max.z), SIMD3(min.x, min.y, max.z),
            SIMD3(min.x, max.y, min.z), SIMD3(max.x, max.y, min.z),
            SIMD3(max.x, max.y, max.z), SIMD3(min.x, max.y, max.z),
        ]
        drawLine(from: c[0], to: c[1], color: color)
        drawLine(from: c[1], to: c[2], color: color)
        drawLine(from: c[2], to: c[3], color: color)
        drawLine(from: c[3], to: c[0], color: color)
        drawLine(from: c[4], to: c[5], color: color)
        drawLine(from: c[5], to: c[6], color: color)
        drawLine(from: c[6], to: c[7], color: color)
        drawLine(from: c[7], to: c[4], color: color)
        drawLine(from: c[0], to: c[4], color: color)
        drawLine(from: c[1], to: c[5], color: color)
        drawLine(from: c[2], to: c[6], color: color)
        drawLine(from: c[3], to: c[7], color: color)
    }

    /// Wireframe cube centered at `center` with uniform `halfExtent`.
    func drawCube(center: SIMD3<Float>, halfExtent: Float, color: SIMD4<Float> = SIMD4(1, 1, 1, 1))
    {
        let h = SIMD3<Float>(repeating: halfExtent)
        drawBox(min: center - h, max: center + h, color: color)
    }

    // MARK: - Sphere

    /// Wireframe sphere: three orthogonal great circles.
    func drawSphere(
        center: SIMD3<Float>, radius: Float,
        color: SIMD4<Float> = SIMD4(1, 1, 1, 1), segments: Int = 32
    ) {
        drawCircle(
            center: center, normal: SIMD3(1, 0, 0), radius: radius, color: color, segments: segments
        )
        drawCircle(
            center: center, normal: SIMD3(0, 1, 0), radius: radius, color: color, segments: segments
        )
        drawCircle(
            center: center, normal: SIMD3(0, 0, 1), radius: radius, color: color, segments: segments
        )
    }

    // MARK: - Capsule

    /// Wireframe capsule between `start` and `end` with the given `radius`.
    func drawCapsule(
        start: SIMD3<Float>, end: SIMD3<Float>, radius: Float,
        color: SIMD4<Float> = SIMD4(1, 1, 1, 1), segments: Int = 16
    ) {
        let dir = end - start
        let len = length(dir)
        guard len > 0 else {
            drawSphere(center: start, radius: radius, color: color, segments: segments)
            return
        }
        let axis = dir / len
        let (tangent, bitangent) = basisForNormal(axis)

        // Cylinder body
        drawCircle(center: start, normal: axis, radius: radius, color: color, segments: segments)
        drawCircle(center: end, normal: axis, radius: radius, color: color, segments: segments)
        for t in [tangent, -tangent, bitangent, -bitangent] {
            drawLine(from: start + t * radius, to: end + t * radius, color: color)
        }

        // Hemispheres: two orthogonal semi-arcs per cap
        let halfSegs = max(segments / 2, 4)
        drawArc(
            center: start, axis1: tangent, axis2: -axis, radius: radius,
            from: 0, to: .pi, color: color, segments: halfSegs)
        drawArc(
            center: start, axis1: bitangent, axis2: -axis, radius: radius,
            from: 0, to: .pi, color: color, segments: halfSegs)
        drawArc(
            center: end, axis1: tangent, axis2: axis, radius: radius,
            from: 0, to: .pi, color: color, segments: halfSegs)
        drawArc(
            center: end, axis1: bitangent, axis2: axis, radius: radius,
            from: 0, to: .pi, color: color, segments: halfSegs)
    }

    // MARK: - Frustum

    /// Wireframe frustum from an inverse view-projection matrix (NDC → world).
    func drawFrustum(
        viewProjectionInverse: simd_float4x4,
        color: SIMD4<Float> = SIMD4(1, 1, 1, 1)
    ) {
        func unproject(_ ndc: SIMD4<Float>) -> SIMD3<Float> {
            let v = viewProjectionInverse * ndc
            return SIMD3(v.x, v.y, v.z) / v.w
        }
        let nbl = unproject(SIMD4(-1, -1, -1, 1))
        let nbr = unproject(SIMD4(1, -1, -1, 1))
        let ntr = unproject(SIMD4(1, 1, -1, 1))
        let ntl = unproject(SIMD4(-1, 1, -1, 1))
        let fbl = unproject(SIMD4(-1, -1, 1, 1))
        let fbr = unproject(SIMD4(1, -1, 1, 1))
        let ftr = unproject(SIMD4(1, 1, 1, 1))
        let ftl = unproject(SIMD4(-1, 1, 1, 1))

        // Near plane
        drawLine(from: nbl, to: nbr, color: color)
        drawLine(from: nbr, to: ntr, color: color)
        drawLine(from: ntr, to: ntl, color: color)
        drawLine(from: ntl, to: nbl, color: color)
        // Far plane
        drawLine(from: fbl, to: fbr, color: color)
        drawLine(from: fbr, to: ftr, color: color)
        drawLine(from: ftr, to: ftl, color: color)
        drawLine(from: ftl, to: fbl, color: color)
        // Connecting edges
        drawLine(from: nbl, to: fbl, color: color)
        drawLine(from: nbr, to: fbr, color: color)
        drawLine(from: ntr, to: ftr, color: color)
        drawLine(from: ntl, to: ftl, color: color)
    }

    // MARK: - Arrow

    /// Arrow with a shaft and a small cone head at `to`.
    func drawArrow(from: SIMD3<Float>, to: SIMD3<Float>, color: SIMD4<Float> = SIMD4(1, 1, 1, 1)) {
        let dir = to - from
        let len = length(dir)
        guard len > 0 else { return }
        let axis = dir / len

        let headLength = len * 0.2
        let headRadius = headLength * 0.4
        let headBase = to - axis * headLength

        drawLine(from: from, to: headBase, color: color)
        drawCircle(center: headBase, normal: axis, radius: headRadius, color: color, segments: 16)

        let (tangent, bitangent) = basisForNormal(axis)
        for t in [tangent, -tangent, bitangent, -bitangent] {
            drawLine(from: to, to: headBase + t * headRadius, color: color)
        }
    }

    // MARK: - Ring

    /// A circle (ring) in the plane defined by `normal`.
    func drawRing(
        center: SIMD3<Float>, normal: SIMD3<Float>, radius: Float,
        color: SIMD4<Float> = SIMD4(1, 1, 1, 1), segments: Int = 32
    ) {
        drawCircle(center: center, normal: normal, radius: radius, color: color, segments: segments)
    }

    // MARK: - Quad

    /// Wireframe quad defined by four corners (in order).
    func drawQuad(
        _ v0: SIMD3<Float>, _ v1: SIMD3<Float>,
        _ v2: SIMD3<Float>, _ v3: SIMD3<Float>,
        color: SIMD4<Float> = SIMD4(1, 1, 1, 1)
    ) {
        drawLine(from: v0, to: v1, color: color)
        drawLine(from: v1, to: v2, color: color)
        drawLine(from: v2, to: v3, color: color)
        drawLine(from: v3, to: v0, color: color)
    }

    // MARK: - Cone

    /// Wireframe cone from `tip` to a circular base centred at `base`.
    func drawCone(
        tip: SIMD3<Float>, base: SIMD3<Float>, radius: Float,
        color: SIMD4<Float> = SIMD4(1, 1, 1, 1), segments: Int = 16
    ) {
        let dir = base - tip
        let len = length(dir)
        guard len > 0 else { return }
        let axis = dir / len

        drawCircle(center: base, normal: axis, radius: radius, color: color, segments: segments)

        let (tangent, bitangent) = basisForNormal(axis)
        for t in [tangent, -tangent, bitangent, -bitangent] {
            drawLine(from: tip, to: base + t * radius, color: color)
        }
    }

    // MARK: - Cylinder

    /// Wireframe cylinder between `start` and `end`.
    func drawCylinder(
        start: SIMD3<Float>, end: SIMD3<Float>, radius: Float,
        color: SIMD4<Float> = SIMD4(1, 1, 1, 1), segments: Int = 16
    ) {
        let dir = end - start
        let len = length(dir)
        guard len > 0 else { return }
        let axis = dir / len

        drawCircle(center: start, normal: axis, radius: radius, color: color, segments: segments)
        drawCircle(center: end, normal: axis, radius: radius, color: color, segments: segments)

        let (tangent, bitangent) = basisForNormal(axis)
        for t in [tangent, -tangent, bitangent, -bitangent] {
            drawLine(from: start + t * radius, to: end + t * radius, color: color)
        }
    }

    // MARK: - Axes

    /// RGB XYZ axis gizmo at `origin`.
    func drawAxes(origin: SIMD3<Float>, size: Float = 1.0) {
        drawLine(from: origin, to: origin + SIMD3(size, 0, 0), color: SIMD4(1, 0, 0, 1))
        drawLine(from: origin, to: origin + SIMD3(0, size, 0), color: SIMD4(0, 1, 0, 1))
        drawLine(from: origin, to: origin + SIMD3(0, 0, size), color: SIMD4(0, 0, 1, 1))
    }

    // MARK: - Geometry helpers (private)

    /// Returns two orthogonal tangent vectors spanning the plane perpendicular to `normal`.
    private func basisForNormal(_ normal: SIMD3<Float>) -> (SIMD3<Float>, SIMD3<Float>) {
        let up: SIMD3<Float> =
            abs(dot(normal, SIMD3(0, 1, 0))) < 0.99
            ? SIMD3(0, 1, 0) : SIMD3(1, 0, 0)
        let tangent = normalize(cross(normal, up))
        let bitangent = normalize(cross(normal, tangent))
        return (tangent, bitangent)
    }

    /// Draws a full circle in the plane perpendicular to `normal`.
    private func drawCircle(
        center: SIMD3<Float>, normal: SIMD3<Float>, radius: Float,
        color: SIMD4<Float>, segments: Int
    ) {
        let (t, b) = basisForNormal(normal)
        let step = 2.0 * Float.pi / Float(segments)
        for i in 0..<segments {
            let a0 = Float(i) * step
            let a1 = Float(i + 1) * step
            let p0 = center + radius * (cos(a0) * t + sin(a0) * b)
            let p1 = center + radius * (cos(a1) * t + sin(a1) * b)
            drawLine(from: p0, to: p1, color: color)
        }
    }

    /// Draws an arc in the plane spanned by `axis1` and `axis2`.
    private func drawArc(
        center: SIMD3<Float>, axis1: SIMD3<Float>, axis2: SIMD3<Float>,
        radius: Float, from startAngle: Float, to endAngle: Float,
        color: SIMD4<Float>, segments: Int
    ) {
        let step = (endAngle - startAngle) / Float(segments)
        for i in 0..<segments {
            let a0 = startAngle + Float(i) * step
            let a1 = startAngle + Float(i + 1) * step
            let p0 = center + radius * (cos(a0) * axis1 + sin(a0) * axis2)
            let p1 = center + radius * (cos(a1) * axis1 + sin(a1) * axis2)
            drawLine(from: p0, to: p1, color: color)
        }
    }

    // MARK: - Pass

    override func render(context: FrameContext) {
        defer { vertices.removeAll(keepingCapacity: true) }
        guard context.scene != nil else { return }

        let nextFrame = (context.frameIndex + 1) % 3
        context.sceneBuffer.resetDebugDraw(forFrame: nextFrame)

        // Upload CPU vertices
        if !vertices.isEmpty {
            let buf = vertexBuffers[context.frameIndex]
            let byteSize = vertices.count * MemoryLayout<DebugVertex>.stride
            vertices.withUnsafeBytes { ptr in
                buf.write(bytes: ptr.baseAddress!, size: byteSize)
            }
        }

        let useDepth = settings?.debugDepthTest == true
        let depthTex: Texture? = useDepth ? context.resources.get("GBuffer.Depth") : nil

        var rpDesc = RenderPassDescriptor()
        rpDesc.name = "Debug Pass"
        rpDesc.addAttachment(texture: context.drawable.texture, shouldClear: false)
        if let depth = depthTex {
            rpDesc.setDepthAttachment(texture: depth, shouldClear: false, shouldStore: true)
        }

        var data = DebugData(camera: context.camera.viewProjection)

        let icb = icbs[context.frameIndex]

        // Generate ICB
        let cp = context.cmdBuffer.beginComputePass(name: "Generate Debug ICB")
        cp.consumerBarrier(before: .dispatch, after: [.fragment, .dispatch])
        cp.setPipeline(pipeline: icbPipe)
        cp.setBuffer(buf: context.sceneBuffer.buffer, index: 0)
        cp.setBuffer(buf: icb.buffer, index: 1)
        cp.dispatch(threads: MTLSizeMake(1, 1, 1), threadsPerGroup: MTLSizeMake(1, 1, 1))
        cp.end()

        let rp = context.cmdBuffer.beginRenderPass(descriptor: rpDesc)
        rp.consumerBarrier(before: .vertex, after: .dispatch)
        rp.setPipeline(pipeline: depthTex != nil ? pipelineDepth : pipelineNoDepth)
        rp.setBytes(allocator: context.allocator, index: 0, bytes: &data, size: MemoryLayout<DebugData>.size, stages: .vertex)
        if !vertices.isEmpty {
            rp.pushMarker(name: "CPU Flush")
            rp.setBuffer(buf: vertexBuffers[context.frameIndex], index: 1, stages: .vertex)
            rp.draw(primitiveType: .line, vertexCount: vertices.count, vertexOffset: 0)
            rp.popMarker()
        }

        let gpuBuf = context.sceneBuffer.debugVerticesBuffer(forFrame: context.frameIndex)
        rp.pushMarker(name: "GPU Flush")
        rp.setBuffer(buf: gpuBuf, index: 1, stages: .vertex)
        rp.executeIndirect(icb: icb, maxCommandCount: 1)
        rp.popMarker()
        rp.end()
    }
}
