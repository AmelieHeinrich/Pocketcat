//
//  SettingsView.swift
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 02/03/2026.
//

import SwiftUI

struct SettingsView: View {
    @ObservedObject var settings: RendererSettings

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // Main section
                VStack(alignment: .leading, spacing: 12) {
                    Text("Main Settings")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .textCase(.uppercase)

                    HStack {
                        Text("Timeline")
                            .font(.system(size: 12, weight: .medium))
                        Spacer()
                        Picker("", selection: $settings.currentTimeline) {
                            Text("Mobile").tag(RendererTimelineType.Mobile)
                            Text("Desktop").tag(RendererTimelineType.Desktop)
                            Text("Pathtraced").tag(RendererTimelineType.Pathtraced)
                        }
                        .pickerStyle(.menu)
                        .labelsHidden()
                        .frame(maxWidth: 120)
                    }
                }

                // Forward Pass section
                VStack(alignment: .leading, spacing: 12) {
                    Text("Forward Pass")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .textCase(.uppercase)

                    HStack {
                        Text("Mesh Shaders")
                            .font(.system(size: 12, weight: .medium))
                        Spacer()
                        Toggle("", isOn: $settings.useMeshShader)
                            .toggleStyle(.switch)
                            .labelsHidden()
                    }

                    HStack {
                        Text("Forced LOD")
                            .font(.system(size: 12, weight: .medium))
                        Spacer()
                        Text("LOD\(settings.forcedLOD)")
                            .font(.system(size: 12).monospacedDigit())
                            .foregroundStyle(.secondary)
                        Stepper("", value: $settings.forcedLOD, in: 0...4)
                            .labelsHidden()
                    }
                }

                // Debug Draw section
                VStack(alignment: .leading, spacing: 12) {
                    Text("Debug Draw")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .textCase(.uppercase)

                    HStack {
                        Text("Depth Test")
                            .font(.system(size: 12, weight: .medium))
                        Spacer()
                        Toggle("", isOn: $settings.debugDepthTest)
                            .toggleStyle(.switch)
                            .labelsHidden()
                    }
                }

                // Tonemap section
                VStack(alignment: .leading, spacing: 12) {
                    Text("Tonemap")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .textCase(.uppercase)

                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text("Gamma")
                                .font(.system(size: 12, weight: .medium))
                            Spacer()
                            Text(String(format: "%.2f", settings.tonemapGamma))
                                .font(.system(size: 12).monospacedDigit())
                                .foregroundStyle(.secondary)
                        }
                        Slider(value: $settings.tonemapGamma, in: 1.0...3.0, step: 0.01)
                            .tint(.indigo)
                    }
                }
            }
            .padding(16)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
