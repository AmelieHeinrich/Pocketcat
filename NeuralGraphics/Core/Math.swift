//
//  Math.swift
//  Neural Graphics
//
//  Created by Amélie Heinrich on 28/02/2026.
//

import simd

// MARK: - Float constants

extension Float {
    static let pi = Float.pi
    static func radians(_ degrees: Float) -> Float { degrees * (.pi / 180.0) }
    static func degrees(_ radians: Float) -> Float { radians * (180.0 / .pi) }
}

// MARK: - simd_float4x4 helpers

extension simd_float4x4 {
    static let identity = matrix_identity_float4x4

    static func translation(_ t: SIMD3<Float>) -> simd_float4x4 {
        var m = matrix_identity_float4x4
        m.columns.3 = SIMD4<Float>(t.x, t.y, t.z, 1.0)
        return m
    }

    static func scale(_ s: SIMD3<Float>) -> simd_float4x4 {
        simd_float4x4(diagonal: SIMD4<Float>(s.x, s.y, s.z, 1.0))
    }

    static func rotationX(_ angle: Float) -> simd_float4x4 {
        let c = cos(angle), s = sin(angle)
        return simd_float4x4(columns: (
            SIMD4<Float>(1,  0,  0, 0),
            SIMD4<Float>(0,  c,  s, 0),
            SIMD4<Float>(0, -s,  c, 0),
            SIMD4<Float>(0,  0,  0, 1)
        ))
    }

    static func rotationY(_ angle: Float) -> simd_float4x4 {
        let c = cos(angle), s = sin(angle)
        return simd_float4x4(columns: (
            SIMD4<Float>( c, 0, -s, 0),
            SIMD4<Float>( 0, 1,  0, 0),
            SIMD4<Float>( s, 0,  c, 0),
            SIMD4<Float>( 0, 0,  0, 1)
        ))
    }

    static func rotationZ(_ angle: Float) -> simd_float4x4 {
        let c = cos(angle), s = sin(angle)
        return simd_float4x4(columns: (
            SIMD4<Float>( c, s, 0, 0),
            SIMD4<Float>(-s, c, 0, 0),
            SIMD4<Float>( 0, 0, 1, 0),
            SIMD4<Float>( 0, 0, 0, 1)
        ))
    }

    /// Right-handed perspective projection (depth range 0…1, Metal convention)
    static func perspectiveRH(fovY: Float, aspect: Float, near: Float, far: Float) -> simd_float4x4 {
        let y = 1.0 / tan(fovY * 0.5)
        let x = y / aspect
        let z = far / (near - far)
        return simd_float4x4(columns: (
            SIMD4<Float>(x,  0,  0,  0),
            SIMD4<Float>(0,  y,  0,  0),
            SIMD4<Float>(0,  0,  z, -1),
            SIMD4<Float>(0,  0,  z * near, 0)
        ))
    }

    /// Right-handed look-at view matrix
    static func lookAtRH(eye: SIMD3<Float>, center: SIMD3<Float>, up: SIMD3<Float>) -> simd_float4x4 {
        let f = normalize(center - eye)   // forward
        let r = normalize(cross(f, up))   // right
        let u = cross(r, f)              // up (re-orthogonalized)
        return simd_float4x4(columns: (
            SIMD4<Float>( r.x,  u.x, -f.x, 0),
            SIMD4<Float>( r.y,  u.y, -f.y, 0),
            SIMD4<Float>( r.z,  u.z, -f.z, 0),
            SIMD4<Float>(-dot(r, eye), -dot(u, eye), dot(f, eye), 1)
        ))
    }
}
