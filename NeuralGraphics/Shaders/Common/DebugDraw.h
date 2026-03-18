//
//  DebugDraw.h
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 2026.
//
//  GPU-side debug draw primitives. Include this header from any shader that
//  has access to a `device SceneBuffer*` and wants to emit debug geometry.
//
//  All functions are thread-safe: vertex slots are claimed with an atomic add,
//  so multiple threads can call these simultaneously without races.
//
//  Usage:
//    #include "Common/DebugDraw.h"
//
//    kernel void myKernel(device SceneBuffer* scene [[buffer(0)]], ...) {
//        debug_draw_box(scene, aabbMin, aabbMax, float4(1, 0, 0, 1));
//        debug_draw_box(scene, aabbMin, aabbMax, float4(1, 0, 0, 1), myTransform);
//    }
//

#ifndef DEBUG_DRAW_H
#define DEBUG_DRAW_H

#include "Bindless.h"

// ---------------------------------------------------------------------------
// Internal helpers — not part of the public API
// ---------------------------------------------------------------------------

/// Appends one line segment (two vertices) to the GPU debug vertex buffer.
/// If `transform` is not identity the endpoints are transformed before writing.
static inline void _debug_line(device SceneBuffer* scene,
                                float3 from, float3 to, float4 color,
                                float4x4 transform = float4x4(1.0f))
{
    from = (transform * float4(from, 1.0f)).xyz;
    to   = (transform * float4(to,   1.0f)).xyz;

    uint idx = atomic_fetch_add_explicit(scene->DebugVertexCount, 2u,
                                         memory_order_relaxed);
    if (idx + 1u >= scene->MaxDebugVertices) return;
    scene->DebugVertices[idx]     = { packed_float3(from), packed_float4(color) };
    scene->DebugVertices[idx + 1] = { packed_float3(to),   packed_float4(color) };
}

/// Returns two orthogonal tangent vectors perpendicular to `n`.
static inline void _debug_basis(float3 n, thread float3& t, thread float3& b)
{
    float3 up = (abs(dot(n, float3(0, 1, 0))) < 0.99f) ? float3(0, 1, 0) : float3(1, 0, 0);
    t = normalize(cross(n, up));
    b = normalize(cross(n, t));
}

/// Draws a full circle in the plane perpendicular to `normal`.
static inline void _debug_circle(device SceneBuffer* scene,
                                  float3 center, float3 normal, float radius,
                                  float4 color, int segments,
                                  float4x4 transform = float4x4(1.0f))
{
    float3 t, b;
    _debug_basis(normal, t, b);
    float step = 2.0f * M_PI_F / float(segments);
    for (int i = 0; i < segments; ++i) {
        float a0 = float(i)     * step;
        float a1 = float(i + 1) * step;
        float3 p0 = center + radius * (cos(a0) * t + sin(a0) * b);
        float3 p1 = center + radius * (cos(a1) * t + sin(a1) * b);
        _debug_line(scene, p0, p1, color, transform);
    }
}

/// Draws an arc in the plane spanned by `axis1` and `axis2`.
static inline void _debug_arc(device SceneBuffer* scene,
                               float3 center, float3 axis1, float3 axis2,
                               float radius, float startAngle, float endAngle,
                               float4 color, int segments,
                               float4x4 transform = float4x4(1.0f))
{
    float step = (endAngle - startAngle) / float(segments);
    for (int i = 0; i < segments; ++i) {
        float a0 = startAngle + float(i)     * step;
        float a1 = startAngle + float(i + 1) * step;
        float3 p0 = center + radius * (cos(a0) * axis1 + sin(a0) * axis2);
        float3 p1 = center + radius * (cos(a1) * axis1 + sin(a1) * axis2);
        _debug_line(scene, p0, p1, color, transform);
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Draw a single line segment.
inline void debug_draw_line(device SceneBuffer* scene,
                             float3 from, float3 to,
                             float4 color = float4(1),
                             float4x4 transform = float4x4(1.0f))
{
    _debug_line(scene, from, to, color, transform);
}

/// Wireframe axis-aligned box from `bmin` to `bmax`.
inline void debug_draw_box(device SceneBuffer* scene,
                            float3 bmin, float3 bmax,
                            float4 color = float4(1),
                            float4x4 transform = float4x4(1.0f))
{
    float3 c[8] = {
        float3(bmin.x, bmin.y, bmin.z), float3(bmax.x, bmin.y, bmin.z),
        float3(bmax.x, bmin.y, bmax.z), float3(bmin.x, bmin.y, bmax.z),
        float3(bmin.x, bmax.y, bmin.z), float3(bmax.x, bmax.y, bmin.z),
        float3(bmax.x, bmax.y, bmax.z), float3(bmin.x, bmax.y, bmax.z),
    };
    // Bottom face
    _debug_line(scene, c[0], c[1], color, transform); _debug_line(scene, c[1], c[2], color, transform);
    _debug_line(scene, c[2], c[3], color, transform); _debug_line(scene, c[3], c[0], color, transform);
    // Top face
    _debug_line(scene, c[4], c[5], color, transform); _debug_line(scene, c[5], c[6], color, transform);
    _debug_line(scene, c[6], c[7], color, transform); _debug_line(scene, c[7], c[4], color, transform);
    // Verticals
    _debug_line(scene, c[0], c[4], color, transform); _debug_line(scene, c[1], c[5], color, transform);
    _debug_line(scene, c[2], c[6], color, transform); _debug_line(scene, c[3], c[7], color, transform);
}

/// Wireframe cube centred at `center` with uniform `halfExtent`.
inline void debug_draw_cube(device SceneBuffer* scene,
                             float3 center, float halfExtent,
                             float4 color = float4(1),
                             float4x4 transform = float4x4(1.0f))
{
    float3 h = float3(halfExtent);
    debug_draw_box(scene, center - h, center + h, color, transform);
}

/// Wireframe sphere: three orthogonal great circles.
inline void debug_draw_sphere(device SceneBuffer* scene,
                               float3 center, float radius,
                               float4 color = float4(1), int segments = 32,
                               float4x4 transform = float4x4(1.0f))
{
    _debug_circle(scene, center, float3(1, 0, 0), radius, color, segments, transform);
    _debug_circle(scene, center, float3(0, 1, 0), radius, color, segments, transform);
    _debug_circle(scene, center, float3(0, 0, 1), radius, color, segments, transform);
}

/// Wireframe capsule between `start` and `end` with the given `radius`.
inline void debug_draw_capsule(device SceneBuffer* scene,
                                float3 start, float3 end, float radius,
                                float4 color = float4(1), int segments = 16,
                                float4x4 transform = float4x4(1.0f))
{
    float3 dir = end - start;
    float  len = length(dir);
    if (len <= 0.0f) { debug_draw_sphere(scene, start, radius, color, segments, transform); return; }
    float3 axis = dir / len;
    float3 tangent, bitangent;
    _debug_basis(axis, tangent, bitangent);

    _debug_circle(scene, start, axis, radius, color, segments, transform);
    _debug_circle(scene, end,   axis, radius, color, segments, transform);

    float3 dirs[4] = { tangent, -tangent, bitangent, -bitangent };
    for (int i = 0; i < 4; ++i)
        _debug_line(scene, start + dirs[i] * radius, end + dirs[i] * radius, color, transform);

    int halfSegs = max(segments / 2, 4);
    _debug_arc(scene, start, tangent,    -axis, radius, 0, M_PI_F, color, halfSegs, transform);
    _debug_arc(scene, start, bitangent,  -axis, radius, 0, M_PI_F, color, halfSegs, transform);
    _debug_arc(scene, end,   tangent,     axis, radius, 0, M_PI_F, color, halfSegs, transform);
    _debug_arc(scene, end,   bitangent,   axis, radius, 0, M_PI_F, color, halfSegs, transform);
}

/// Wireframe frustum from an inverse view-projection matrix (NDC → world).
inline void debug_draw_frustum(device SceneBuffer* scene,
                                float4x4 invViewProj,
                                float4 color = float4(1),
                                float4x4 transform = float4x4(1.0f))
{
    auto unproject = [&](float4 ndc) -> float3 {
        float4 v = invViewProj * ndc;
        return v.xyz / v.w;
    };
    float3 nbl = unproject(float4(-1, -1, -1, 1)), nbr = unproject(float4( 1, -1, -1, 1));
    float3 ntr = unproject(float4( 1,  1, -1, 1)), ntl = unproject(float4(-1,  1, -1, 1));
    float3 fbl = unproject(float4(-1, -1,  1, 1)), fbr = unproject(float4( 1, -1,  1, 1));
    float3 ftr = unproject(float4( 1,  1,  1, 1)), ftl = unproject(float4(-1,  1,  1, 1));

    // Near
    _debug_line(scene, nbl, nbr, color, transform); _debug_line(scene, nbr, ntr, color, transform);
    _debug_line(scene, ntr, ntl, color, transform); _debug_line(scene, ntl, nbl, color, transform);
    // Far
    _debug_line(scene, fbl, fbr, color, transform); _debug_line(scene, fbr, ftr, color, transform);
    _debug_line(scene, ftr, ftl, color, transform); _debug_line(scene, ftl, fbl, color, transform);
    // Edges
    _debug_line(scene, nbl, fbl, color, transform); _debug_line(scene, nbr, fbr, color, transform);
    _debug_line(scene, ntr, ftr, color, transform); _debug_line(scene, ntl, ftl, color, transform);
}

/// Arrow with shaft and small cone head at `to`.
inline void debug_draw_arrow(device SceneBuffer* scene,
                              float3 from, float3 to,
                              float4 color = float4(1),
                              float4x4 transform = float4x4(1.0f))
{
    float3 dir = to - from;
    float  len = length(dir);
    if (len <= 0.0f) return;
    float3 axis = dir / len;

    float  headLen    = len * 0.2f;
    float  headRadius = headLen * 0.4f;
    float3 headBase   = to - axis * headLen;

    _debug_line(scene, from, headBase, color, transform);
    _debug_circle(scene, headBase, axis, headRadius, color, 16, transform);

    float3 tangent, bitangent;
    _debug_basis(axis, tangent, bitangent);
    float3 dirs[4] = { tangent, -tangent, bitangent, -bitangent };
    for (int i = 0; i < 4; ++i)
        _debug_line(scene, to, headBase + dirs[i] * headRadius, color, transform);
}

/// A circle (ring) in the plane defined by `normal`.
inline void debug_draw_ring(device SceneBuffer* scene,
                             float3 center, float3 normal, float radius,
                             float4 color = float4(1), int segments = 32,
                             float4x4 transform = float4x4(1.0f))
{
    _debug_circle(scene, center, normal, radius, color, segments, transform);
}

/// Wireframe quad defined by four corners (in order).
inline void debug_draw_quad(device SceneBuffer* scene,
                             float3 v0, float3 v1, float3 v2, float3 v3,
                             float4 color = float4(1),
                             float4x4 transform = float4x4(1.0f))
{
    _debug_line(scene, v0, v1, color, transform); _debug_line(scene, v1, v2, color, transform);
    _debug_line(scene, v2, v3, color, transform); _debug_line(scene, v3, v0, color, transform);
}

/// Wireframe cone from `tip` to a circular base centred at `base`.
inline void debug_draw_cone(device SceneBuffer* scene,
                             float3 tip, float3 base, float radius,
                             float4 color = float4(1), int segments = 16,
                             float4x4 transform = float4x4(1.0f))
{
    float3 dir = base - tip;
    float  len = length(dir);
    if (len <= 0.0f) return;
    float3 axis = dir / len;

    _debug_circle(scene, base, axis, radius, color, segments, transform);

    float3 tangent, bitangent;
    _debug_basis(axis, tangent, bitangent);
    float3 dirs[4] = { tangent, -tangent, bitangent, -bitangent };
    for (int i = 0; i < 4; ++i)
        _debug_line(scene, tip, base + dirs[i] * radius, color, transform);
}

/// Wireframe cylinder between `start` and `end`.
inline void debug_draw_cylinder(device SceneBuffer* scene,
                                 float3 start, float3 end, float radius,
                                 float4 color = float4(1), int segments = 16,
                                 float4x4 transform = float4x4(1.0f))
{
    float3 dir = end - start;
    float  len = length(dir);
    if (len <= 0.0f) return;
    float3 axis = dir / len;

    _debug_circle(scene, start, axis, radius, color, segments, transform);
    _debug_circle(scene, end,   axis, radius, color, segments, transform);

    float3 tangent, bitangent;
    _debug_basis(axis, tangent, bitangent);
    float3 dirs[4] = { tangent, -tangent, bitangent, -bitangent };
    for (int i = 0; i < 4; ++i)
        _debug_line(scene, start + dirs[i] * radius, end + dirs[i] * radius, color, transform);
}

/// RGB XYZ axis gizmo at `origin` (X=red, Y=green, Z=blue).
inline void debug_draw_axes(device SceneBuffer* scene,
                             float3 origin, float size = 1.0f,
                             float4x4 transform = float4x4(1.0f))
{
    _debug_line(scene, origin, origin + float3(size, 0, 0), float4(1, 0, 0, 1), transform);
    _debug_line(scene, origin, origin + float3(0, size, 0), float4(0, 1, 0, 1), transform);
    _debug_line(scene, origin, origin + float3(0, 0, size), float4(0, 0, 1, 1), transform);
}

#endif // DEBUG_DRAW_H
