//
//  RendererSettings.swift
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 02/03/2026.
//

import Combine
import Foundation

enum RendererTimelineType {
    case Mobile  // Forward, no raytracing
    case Desktop  // Deferred, full raytracing
    case Pathtraced  // Pathtraced reference
}

class RendererSettings: ObservableObject {
    @Published var currentTimeline: RendererTimelineType = .Desktop
    @Published var tonemapGamma: Float = 2.2
    @Published var debugDepthTest: Bool = false
    @Published var useMeshShader: Bool = true
    @Published var forcedLOD: Int = 0
}
