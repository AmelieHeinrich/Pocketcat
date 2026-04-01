//
//  Scene.swift
//  Pocketcat
//
//  Created by Amélie Heinrich on 06/03/2026.
//

import simd
import Foundation

// MARK: - Entity

/// A renderable object in the scene: a loaded Mesh plus a model-space transform.
struct Entity {
    var mesh: Mesh
    var transform: simd_float4x4 = .identity
}

// MARK: - RenderScene

/// Holds all the entities that make up a loaded scene.
class RenderScene {
    var entities: [Entity] = []
    var tlas: TLAS

    init() {
        tlas = TLAS(makeResidentNow: true)
        tlas.setName(name: "Scene TLAS")
    }
}

// MARK: - Descriptors (data layer — no SwiftUI)

/// Describes one model to load: its bundle resource name and an initial transform.
struct SceneModelDescriptor {
    var resource: String  // bundle resource name without extension (.bin implied)
    var transform: simd_float4x4 = .identity
}

/// Describes an entire scene: the ordered list of models to load.
struct SceneDescriptor {
    var models: [SceneModelDescriptor]
}

// MARK: - Scene Group

enum SceneGroup: String, CaseIterable {
    case showcase       = "Showcase"
    case apiPerformance = "API / Performance"
}

// MARK: - Scene Configuration (preset)

struct SceneConfiguration: Identifiable {
    let id: String
    let name: String
    let systemIcon: String
    let tags: [String]
    let group: SceneGroup
    let descriptor: SceneDescriptor

    var isAvailable: Bool {
        guard let first = descriptor.models.first else { return true }
        return Bundle.main.url(forResource: first.resource, withExtension: "bin") != nil
    }
}

// MARK: - Built-in presets

extension SceneConfiguration {
    static let all: [SceneConfiguration] = [
        SceneConfiguration(
            id: "cube",
            name: "Cube",
            systemIcon: "cube",
            tags: ["Basic"],
            group: .showcase,
            descriptor: SceneDescriptor(models: [
                SceneModelDescriptor(resource: "Cube")
            ])
        ),
        SceneConfiguration(
            id: "sponza",
            name: "Sponza",
            systemIcon: "building.columns",
            tags: ["Classic"],
            group: .showcase,
            descriptor: SceneDescriptor(models: [
                SceneModelDescriptor(resource: "Sponza")
            ])
        ),
        SceneConfiguration(
            id: "bistro",
            name: "Bistro",
            systemIcon: "house",
            tags: ["Exterior", "Complex"],
            group: .showcase,
            descriptor: SceneDescriptor(models: [
                SceneModelDescriptor(resource: "bistro_ext")
            ])
        ),
        SceneConfiguration(
            id: "intel_sponza",
            name: "Intel Sponza",
            systemIcon: "building.columns.fill",
            tags: ["PBR"],
            group: .showcase,
            descriptor: SceneDescriptor(models: [
                SceneModelDescriptor(resource: "IntelSponza")
            ])
        ),
        SceneConfiguration(
            id: "cube_storm",
            name: "Cube Storm",
            systemIcon: "square.grid.3x3.fill",
            tags: ["Stress test"],
            group: .apiPerformance,
            descriptor: SceneDescriptor(
                models: (0..<8092).map { _ in
                    SceneModelDescriptor(
                        resource: "Cube",
                        transform: simd_float4x4.translation(
                            simd_float3(
                                Float.random(in: -200...200),
                                Float.random(in: 0...50),
                                Float.random(in: -200...200)
                            ))
                    )
                })
        ),
        SceneConfiguration(
            id: "buddha",
            name: "Buddha",
            systemIcon: "figure.mind.and.body",
            tags: ["Basic"],
            group: .showcase,
            descriptor: SceneDescriptor(models: [
                SceneModelDescriptor(resource: "Buddha")
            ])
        ),
        SceneConfiguration(
            id: "cube_sphere",
            name: "Cube And Sphere",
            systemIcon: "cube.circle",
            tags: ["RT Test"],
            group: .apiPerformance,
            descriptor: SceneDescriptor(models: [
                SceneModelDescriptor(resource: "CubeAndSphere")
            ])
        ),
        SceneConfiguration(
            id: "buddha_storm",
            name: "Buddha Storm",
            systemIcon: "sparkles",
            tags: ["Stress test", "Mesh shaders"],
            group: .apiPerformance,
            descriptor: SceneDescriptor(
                models: (0..<64).map { _ in
                    SceneModelDescriptor(
                        resource: "Buddha",
                        transform: simd_float4x4.translation(
                            simd_float3(
                                Float.random(in: -5...5),
                                Float.random(in: -5...5),
                                Float.random(in: -5...5)
                            )) * simd_float4x4.rotationX(Float.random(in: -180...180))
                            * simd_float4x4.rotationY(Float.random(in: -180...180))
                            * simd_float4x4.rotationZ(Float.random(in: -180...180))
                    )
                })
        ),
        SceneConfiguration(
            id: "shadow_test",
            name: "Shadow Test",
            systemIcon: "sun.horizon",
            tags: ["RT Test"],
            group: .apiPerformance,
            descriptor: SceneDescriptor(models: [
                SceneModelDescriptor(resource: "ShadowTest")
            ])
        ),
    ]
}
