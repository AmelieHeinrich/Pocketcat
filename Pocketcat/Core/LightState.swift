//
//  LightState.swift
//  Pocketcat
//
//  Created by Amélie Heinrich on 2026.
//

import Foundation
import simd
import SwiftUI
import Combine

enum SunMode: String, CaseIterable {
    case manual = "Manual"
    case timeOfDay = "Time of Day"
}

struct PointLightEntry: Identifiable {
    let id: UUID
    var position: SIMD3<Float>
    var radius: Float
    var intensity: Float
    var color: Color

    init(position: SIMD3<Float> = .zero, radius: Float = 5.0, intensity: Float = 5.0, color: Color = .white) {
        self.id = UUID()
        self.position = position
        self.radius = radius
        self.intensity = intensity
        self.color = color
    }
}

class LightState: ObservableObject {
    @Published var sunMode: SunMode = .manual

    // Manual sun control
    @Published var sunAzimuth: Float = 224.0
    @Published var sunElevation: Float = 75.0

    // Shared sun properties
    @Published var sunColor: Color = Color(red: 1.0, green: 0.95, blue: 0.85)
    @Published var sunIntensity: Float = 3.0
    @Published var sunRadius: Float = 0.05     // angular radius for soft shadows

    // Point lights
    @Published var pointLights: [PointLightEntry] = []

    // MARK: - Computed sun direction

    var sunDirection: SIMD3<Float> {
        switch sunMode {
        case .manual:
            return directionFromAzEl(azimuth: sunAzimuth, elevation: sunElevation)
        case .timeOfDay:
            return computeSunFromTime(Date()).direction
        }
    }

    // MARK: - GPU Data

    func gpuSun() -> GPUSunLight {
        let dir: SIMD3<Float>
        let col: SIMD3<Float>
        let intensity: Float

        switch sunMode {
        case .manual:
            dir = directionFromAzEl(azimuth: sunAzimuth, elevation: sunElevation)
            col = colorToFloat3(sunColor)
            intensity = sunIntensity
        case .timeOfDay:
            let sun = computeSunFromTime(Date())
            dir = sun.direction
            col = sun.color
            intensity = sun.intensity
        }

        return GPUSunLight(
            directionAndRadius: SIMD4<Float>(dir, sunRadius),
            colorAndIntensity: SIMD4<Float>(col, intensity))
    }

    func gpuPointLights() -> [GPUPointLight] {
        return pointLights.map { light in
            let col = colorToFloat3(light.color)
            return GPUPointLight(
                positionAndRadius: SIMD4<Float>(light.position, light.radius),
                colorAndIntensity: SIMD4<Float>(col, light.intensity))
        }
    }

    // MARK: - Private helpers

    private func directionFromAzEl(azimuth: Float, elevation: Float) -> SIMD3<Float> {
        let az = azimuth * .pi / 180.0
        let el = elevation * .pi / 180.0
        // Returns direction pointing FROM the sun TOWARD the scene (used as light_dir in shaders)
        return normalize(SIMD3<Float>(
            cos(el) * sin(az),
            -sin(el),
            cos(el) * cos(az)
        ))
    }

    private func computeSunFromTime(_ date: Date) -> (direction: SIMD3<Float>, color: SIMD3<Float>, intensity: Float) {
        let cal = Calendar.current
        let hour = Float(cal.component(.hour, from: date))
        let minute = Float(cal.component(.minute, from: date))
        let t = (hour + minute / 60.0) / 24.0  // 0..1 over the day

        // Solar elevation: peaks at noon (t=0.5), deeply negative at midnight
        // sin curve shifted so t=0.25 is sunrise (0°), t=0.5 is noon (80°), t=0.75 is sunset (0°)
        let solarAngle = (t - 0.25) * 2.0 * Float.pi
        let elevation = sin(solarAngle) * 80.0  // degrees

        // Azimuth sweeps east→south→west over the day
        let azimuth = 90.0 + t * 360.0  // rises in east, sets in west

        let direction = directionFromAzEl(azimuth: azimuth, elevation: elevation)

        // Color temperature by elevation:
        // below horizon → black, sunrise/sunset → deep red-orange,
        // golden hour (~5–20°) → warm gold, daytime → near-white
        let color: SIMD3<Float>
        if elevation <= 0 {
            color = SIMD3<Float>(0, 0, 0)
        } else if elevation < 5 {
            let f = elevation / 5.0
            color = SIMD3<Float>(1.0, mix(0.25, 0.55, f), mix(0.0, 0.05, f))
        } else if elevation < 20 {
            let f = (elevation - 5.0) / 15.0
            color = SIMD3<Float>(1.0, mix(0.55, 0.88, f), mix(0.05, 0.65, f))
        } else {
            color = SIMD3<Float>(1.0, 0.95, 0.88)
        }

        let elevRad = elevation * .pi / 180.0
        let intensity = max(0.0, sin(elevRad)) * 5.0

        return (direction: direction, color: color, intensity: intensity)
    }

    private func colorToFloat3(_ color: Color) -> SIMD3<Float> {
        let resolved = color.resolve(in: EnvironmentValues())
        return SIMD3<Float>(resolved.red, resolved.green, resolved.blue)
    }

    private func mix(_ a: Float, _ b: Float, _ t: Float) -> Float {
        return a + (b - a) * t
    }
}
