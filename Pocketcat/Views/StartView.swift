//
//  StartView.swift
//  Pocketcat
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
    "buddha": .teal,
    "cube_sphere": .indigo,
    "buddha_storm": .red,
]

private func color(for config: SceneConfiguration) -> Color {
    configColors[config.id] ?? .white
}

// MARK: - Scene Card

private struct SceneCard: View {
    let config: SceneConfiguration
    let onPick: (SceneConfiguration) -> Void

    @State private var isHovered = false

    private var available: Bool { config.isAvailable }

    var body: some View {
        let accent = available ? color(for: config) : Color.gray

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
                        .foregroundStyle(available ? accent : Color.gray.opacity(0.4))
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

                if !available {
                    HStack(spacing: 4) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.system(size: 7, weight: .semibold))
                            .foregroundStyle(.orange.opacity(0.9))
                        Text("Missing resource")
                            .font(.system(size: 8, weight: .semibold))
                            .textCase(.uppercase)
                            .tracking(0.3)
                            .foregroundStyle(.orange.opacity(0.9))
                    }
                    .padding(.horizontal, 7)
                    .padding(.vertical, 3)
                    .background(Capsule().fill(Color.orange.opacity(0.12)))
                } else if config.tags.isEmpty {
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
                    .fill(.white.opacity(isHovered && available ? 0.08 : 0.04))
                    .overlay(
                        RoundedRectangle(cornerRadius: 18, style: .continuous)
                            .strokeBorder(
                                isHovered && available ? accent.opacity(0.55) : .white.opacity(0.10),
                                lineWidth: 1
                            )
                    )
            )
            .shadow(color: isHovered && available ? accent.opacity(0.25) : .clear, radius: 14, y: 4)
            .contentShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
            .opacity(available ? 1.0 : 0.45)
        }
        .buttonStyle(.plain)
        .disabled(!available)
        .onHover { isHovered = available && $0 }
    }
}

// MARK: - Start View

struct StartView: View {
    let onScenePicked: (SceneConfiguration) -> Void

    @State private var selectedGroup: SceneGroup = .showcase

    private var visibleConfigs: [SceneConfiguration] {
        SceneConfiguration.all.filter { $0.group == selectedGroup }
    }

    private let columns = [
        GridItem(.flexible(), spacing: 20),
        GridItem(.flexible(), spacing: 20),
        GridItem(.flexible(), spacing: 20),
    ]

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            HStack(spacing: 0) {
                // Scene picker
                ScrollView(.vertical, showsIndicators: false) {
                    VStack(spacing: 48) {
                        // Header
                        VStack(spacing: 10) {
                            Image(nsImage: NSApp.applicationIconImage)
                                .resizable()
                                .frame(width: 72, height: 72)

                            Text("Pocketcat")
                                .font(.system(size: 28, weight: .light, design: .rounded))
                                .foregroundStyle(.white)

                            Text("Choose a scene to load")
                                .font(.system(size: 13, weight: .regular))
                                .foregroundStyle(.white.opacity(0.40))

                            Picker("", selection: $selectedGroup) {
                                ForEach(SceneGroup.allCases, id: \.self) { group in
                                    Text(group.rawValue).tag(group)
                                }
                            }
                            .pickerStyle(.segmented)
                            .frame(width: 280)
                            .padding(.top, 6)
                        }

                        // Grid
                        LazyVGrid(columns: columns, spacing: 20) {
                            ForEach(visibleConfigs) { config in
                                SceneCard(config: config, onPick: onScenePicked)
                            }
                        }
                        .frame(maxWidth: 560)
                    }
                    .padding(40)
                    .frame(maxWidth: .infinity)
                }

                // Changelog sidebar
                ChangelogPanel()
            }
        }
    }
}
