//
//  LoadingView.swift
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 02/03/2026.
//

import AppKit
import Combine
import SwiftUI

// MARK: - Scene Loader

/// Loads every model in a `SceneDescriptor` on a background thread and publishes
/// progress back to the main thread so SwiftUI can drive the loading screen.
final class SceneLoader: ObservableObject {
    @Published private(set) var progress: Double = 0
    @Published private(set) var status: String = "Initializing…"
    @Published private(set) var scene: RenderScene? = nil

    var isLoaded: Bool { scene != nil }

    func beginLoading(descriptor: SceneDescriptor) {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let models = descriptor.models

            // Deduplicate: collect the unique resource names in first-seen order.
            var seen = Set<String>()
            var unique = [String]()
            for m in models {
                if seen.insert(m.resource).inserted { unique.append(m.resource) }
            }

            // Load each unique mesh once, reporting progress across unique count.
            var cache = [String: Mesh]()
            for (i, resource) in unique.enumerated() {
                let base = Double(i) / Double(unique.count)
                let slice = 1.0 / Double(unique.count)

                guard let url = Bundle.main.url(forResource: resource, withExtension: "bin") else {
                    DispatchQueue.main.async { [weak self] in
                        self?.status = "Missing: \(resource).bin"
                    }
                    continue
                }

                let mesh = MeshLoader.load(url: url) { p, s in
                    DispatchQueue.main.async { [weak self] in
                        self?.progress = base + p * slice
                        self?.status = "[\(resource)] \(s)"
                    }
                }
                if let mesh { cache[resource] = mesh }
            }

            // Build entities: each descriptor entry maps to its cached mesh.
            let entities: [Entity] = models.compactMap { desc in
                guard let mesh = cache[desc.resource] else { return nil }
                return Entity(mesh: mesh, transform: desc.transform)
            }

            // Make textures resident on the main thread (residency set mutations
            // are not thread-safe). Only do this for the unique meshes.
            DispatchQueue.main.async { [weak self] in
                for mesh in cache.values {
                    for mat in mesh.materials {
                        mat.albedo?.makeResident()
                        mat.normal?.makeResident()
                        mat.orm?.makeResident()
                        mat.emissive?.makeResident()
                    }
                }

                if RendererData.device.supportsFamily(.apple9) {
                    self?.status = "Building Acceleration Structures…"
                    let cb = CommandBuffer()
                    cb.begin()
                    let cp = cb.beginComputePass(name: "Build BLASes")
                    for mesh in cache.values {
                        for blas in mesh.blases {
                            cp.buildBLAS(blas: blas)
                        }
                    }
                    cp.end()
                    cb.end()
                    cb.commit()
                }

                DispatchQueue.global(qos: .userInitiated).async {
                    RendererData.waitIdle()
                    if RendererData.device.supportsFamily(.apple9) {
                        for mesh in cache.values {
                            for blas in mesh.blases {
                                blas.destroyScratch()
                            }
                        }
                    }

                    DispatchQueue.main.async { [weak self] in
                        let scene = RenderScene()
                        scene.entities = entities
                        self?.scene = scene
                        self?.progress = 1.0
                        self?.status = "Ready"
                    }
                }
            }
        }
    }
}

// MARK: - Loading View

struct LoadingView: View {
    let progress: Double
    let status: String

    @State private var pulse = false

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            VStack(spacing: 40) {
                // App icon
                Image(nsImage: NSApp.applicationIconImage)
                    .resizable()
                    .frame(width: 96, height: 96)
                    .scaleEffect(pulse ? 1.06 : 1.0)
                    .animation(
                        .easeInOut(duration: 1.8).repeatForever(autoreverses: true), value: pulse)

                // Title
                VStack(spacing: 6) {
                    Text("Neural Graphics")
                        .font(.system(size: 26, weight: .light, design: .rounded))
                        .foregroundStyle(.white)

                    Text("Loading scene…")
                        .font(.system(size: 12, weight: .regular))
                        .foregroundStyle(.white.opacity(0.40))
                }

                // Progress bar + status
                VStack(spacing: 10) {
                    ZStack(alignment: .leading) {
                        Capsule()
                            .fill(.white.opacity(0.10))
                            .frame(width: 320, height: 4)

                        Capsule()
                            .fill(.white.opacity(0.75))
                            .frame(width: 320 * progress, height: 4)
                            .animation(.easeOut(duration: 0.15), value: progress)
                    }

                    Text(status)
                        .font(.system(size: 11, weight: .regular, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.35))
                        .frame(width: 320, alignment: .leading)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }
        }
        .onAppear { pulse = true }
    }
}
