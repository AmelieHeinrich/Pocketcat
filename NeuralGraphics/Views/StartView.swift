//
//  StartView.swift
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 06/03/2026.
//

import AppKit
import SwiftUI

// MARK: - Palette

private let configColors: [String: Color] = [
    "cube": .cyan,
    "sponza": .orange,
    "bistro": .green,
    "intel_sponza": .purple,
    "cube_storm": .yellow,
    "san_miguel": .mint,
    "living_room": .pink,
    "buddha": .teal,
    "buddha_storm_sponza": .red,
]

private func color(for config: SceneConfiguration) -> Color {
    configColors[config.id] ?? .white
}

// MARK: - Scene Card

private struct SceneCard: View {
    let config: SceneConfiguration
    let onPick: (SceneConfiguration) -> Void

    @State private var isHovered = false

    var body: some View {
        let accent = color(for: config)

        Button {
            onPick(config)
        } label: {
            VStack(spacing: 0) {
                Spacer(minLength: 12)

                ZStack {
                    Circle()
                        .fill(accent.opacity(isHovered ? 0.25 : 0.12))
                        .frame(width: 64, height: 64)
                        .animation(.easeInOut(duration: 0.18), value: isHovered)

                    Image(systemName: config.systemIcon)
                        .font(.system(size: 28, weight: .light))
                        .foregroundStyle(accent)
                        .scaleEffect(isHovered ? 1.12 : 1.0)
                        .animation(.spring(response: 0.3, dampingFraction: 0.6), value: isHovered)
                }

                Spacer(minLength: 10)

                VStack(spacing: 3) {
                    Text(config.name)
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(.white)
                        .lineLimit(1)
                        .minimumScaleFactor(0.75)
                    Text("\(config.descriptor.models.count) model(s)")
                        .font(.system(size: 10, weight: .regular, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.35))
                }

                Spacer(minLength: 8)

                if config.tags.isEmpty {
                    Spacer().frame(height: 20)
                } else {
                    HStack(spacing: 4) {
                        ForEach(config.tags, id: \.self) { tag in
                            Text(tag)
                                .font(.system(size: 8, weight: .semibold))
                                .textCase(.uppercase)
                                .tracking(0.3)
                                .foregroundStyle(accent.opacity(0.85))
                                .padding(.horizontal, 5)
                                .padding(.vertical, 2.5)
                                .background(
                                    Capsule()
                                        .fill(accent.opacity(0.1))
                                )
                        }
                    }
                }

                Spacer(minLength: 10)
            }
            .padding(.horizontal, 8)
            .frame(width: 160, height: 190)
            .background(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .fill(.white.opacity(isHovered ? 0.08 : 0.04))
                    .overlay(
                        RoundedRectangle(cornerRadius: 18, style: .continuous)
                            .strokeBorder(
                                isHovered ? accent.opacity(0.55) : .white.opacity(0.10),
                                lineWidth: 1
                            )
                    )
            )
            .shadow(color: isHovered ? accent.opacity(0.25) : .clear, radius: 14, y: 4)
            .contentShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
        }
        .buttonStyle(.plain)
        .onHover { isHovered = $0 }
    }
}

// MARK: - Start View

struct StartView: View {
    let onScenePicked: (SceneConfiguration) -> Void

    private let configs = SceneConfiguration.all
    private let columns = [
        GridItem(.flexible(), spacing: 20),
        GridItem(.flexible(), spacing: 20),
        GridItem(.flexible(), spacing: 20),
    ]

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            ScrollView(.vertical, showsIndicators: false) {
                VStack(spacing: 48) {
                    // Header
                    VStack(spacing: 10) {
                        Image(nsImage: NSApp.applicationIconImage)
                            .resizable()
                            .frame(width: 72, height: 72)

                        Text("Neural Graphics")
                            .font(.system(size: 28, weight: .light, design: .rounded))
                            .foregroundStyle(.white)

                        Text("Choose a scene to load")
                            .font(.system(size: 13, weight: .regular))
                            .foregroundStyle(.white.opacity(0.40))
                    }

                    // Grid
                    LazyVGrid(columns: columns, spacing: 20) {
                        ForEach(configs) { config in
                            SceneCard(config: config, onPick: onScenePicked)
                        }
                    }
                    .frame(maxWidth: 560)
                }
                .padding(40)
                .frame(maxWidth: .infinity)
            }
        }
    }
}
