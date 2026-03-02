//
//  Camera.swift
//  Neural Graphics
//
//  Created by Amélie Heinrich on 28/02/2026.
//

import simd

class Camera {
    // Transform
    var position: SIMD3<Float>
    var yaw: Float   // horizontal rotation (radians), around Y axis
    var pitch: Float // vertical rotation (radians), around X axis

    // Projection settings
    var fovY: Float
    var near: Float
    var far: Float
    var aspectRatio: Float

    // Movement settings
    var moveSpeed: Float
    var lookSensitivity: Float
    var pitchLimit: Float // max pitch magnitude (radians)

    // Derived vectors (updated each frame)
    private(set) var forward: SIMD3<Float> = .init(0, 0, -1)
    private(set) var right:   SIMD3<Float> = .init(1, 0,  0)
    private(set) var up:      SIMD3<Float> = .init(0, 1,  0)

    // Matrices
    private(set) var viewMatrix:       simd_float4x4 = .identity
    private(set) var projectionMatrix: simd_float4x4 = .identity
    private(set) var viewProjection:   simd_float4x4 = .identity

    init(
        position: SIMD3<Float> = .zero,
        yaw: Float = 0.0,
        pitch: Float = 0.0,
        fovY: Float = Float.radians(60.0),
        aspectRatio: Float = 16.0 / 9.0,
        near: Float = 0.1,
        far: Float = 1000.0,
        moveSpeed: Float = 5.0,
        lookSensitivity: Float = 0.002,
        pitchLimit: Float = Float.radians(89.0)
    ) {
        self.position        = position
        self.yaw             = yaw
        self.pitch           = pitch
        self.fovY            = fovY
        self.aspectRatio     = aspectRatio
        self.near            = near
        self.far             = far
        self.moveSpeed       = moveSpeed
        self.lookSensitivity = lookSensitivity
        self.pitchLimit      = pitchLimit
        updateMatrices()
    }

    // MARK: - Per-frame update

    /// Call once per frame. `dt` is delta time in seconds.
    func update(dt: Float) {
        let input = Input.shared
        handleMovement(dt: dt)
        handleLook(delta: input.mouseDelta)
        updateMatrices()
    }

    // MARK: - Resize

    func resize(width: Float, height: Float) {
        aspectRatio = width / height
        updateMatrices()
    }

    // MARK: - Private helpers

    private func handleMovement(dt: Float) {
        let input = Input.shared
        var velocity: SIMD3<Float> = .zero

        // WASD / arrow keys
        if input.isKeyDown(.w) || input.isKeyDown(.upArrow)    { velocity += forward }
        if input.isKeyDown(.s) || input.isKeyDown(.downArrow)  { velocity -= forward }
        if input.isKeyDown(.d) || input.isKeyDown(.rightArrow) { velocity += right }
        if input.isKeyDown(.a) || input.isKeyDown(.leftArrow)  { velocity -= right }

        // Vertical (Q/E or space/control)
        if input.isKeyDown(.e) || input.isKeyDown(.space)   { velocity.y += 1 }
        if input.isKeyDown(.q) || input.isKeyDown(.control) { velocity.y -= 1 }

        let speed = input.isKeyDown(.shift) ? moveSpeed * 3.0 : moveSpeed
        if simd_length_squared(velocity) > 0 {
            position += normalize(velocity) * speed * dt
        }
    }

    private func handleLook(delta: SIMD2<Float>) {
        guard simd_length_squared(delta) > 0 else { return }
        // Only rotate when right mouse button is held (FPS-style look)
        guard Input.shared.rightMouseDown else { return }

        yaw   += delta.x * lookSensitivity
        pitch -= delta.y * lookSensitivity  // invert Y so moving up looks up
        pitch  = simd_clamp(pitch, -pitchLimit, pitchLimit)
    }

    private func updateMatrices() {
        // Recompute basis vectors from yaw/pitch
        let cp = cos(pitch), sp = sin(pitch)
        let cy = cos(yaw),   sy = sin(yaw)

        forward = normalize(SIMD3<Float>(sy * cp, sp, -cy * cp))
        right   = normalize(cross(forward, SIMD3<Float>(0, 1, 0)))
        up      = cross(right, forward)

        let target = position + forward
        viewMatrix       = .lookAtRH(eye: position, center: target, up: SIMD3<Float>(0, 1, 0))
        projectionMatrix = .perspectiveRH(fovY: fovY, aspect: aspectRatio, near: near, far: far)
        viewProjection   = projectionMatrix * viewMatrix
    }
}
