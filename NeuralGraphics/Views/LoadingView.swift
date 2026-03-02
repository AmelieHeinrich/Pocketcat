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

/// Runs MeshLoader on a background thread and publishes progress back to the
/// main thread so SwiftUI can drive the loading screen.
final class SceneLoader: ObservableObject {
    @Published private(set) var progress: Double = 0
    @Published private(set) var status: String   = "Initializing…"
    @Published private(set) var mesh: Mesh?      = nil

    var isLoaded: Bool { mesh != nil }

    func beginLoading(url: URL) {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let loaded = MeshLoader.load(url: url) { p, s in
                DispatchQueue.main.async { [weak self] in
                    self?.progress = p
                    self?.status   = s
                }
            }
            DispatchQueue.main.async { [weak self] in
                // Make every material texture resident now that we're back on
                // the main thread (residency set mutations are not thread-safe).
                if let mesh = loaded {
                    for mat in mesh.materials {
                        mat.albedo?.makeResident()
                        mat.normal?.makeResident()
                        mat.orm?.makeResident()
                        mat.emissive?.makeResident()
                    }
                }
                self?.mesh     = loaded
                self?.progress = 1.0
                self?.status   = "Ready"
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
                    .animation(.easeInOut(duration: 1.8).repeatForever(autoreverses: true), value: pulse)

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
