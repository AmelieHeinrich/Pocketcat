//
//  ContentView.swift
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 22/02/2026.
//

import SwiftUI
import Metal

// Shared animation used throughout — defined at file scope to avoid
// overload-resolution ambiguity inside closures.
private let panelAnimation: Animation = .spring(response: 0.36, dampingFraction: 0.74)

// MARK: - Panel Side

enum PanelSide {
    case left, right

    var transition: AnyTransition {
        switch self {
        case .left:  return .move(edge: .leading).combined(with: .opacity)
        case .right: return .move(edge: .trailing).combined(with: .opacity)
        }
    }
}

// MARK: - Panel Definition

struct PanelDef: Identifiable {
    let id: String
    let icon: String
    let label: String
    let color: Color
    let side: PanelSide
}

// MARK: - All panels, in display order per side

private let allPanels: [PanelDef] = [
    // Left column — top to bottom
    PanelDef(id: "training", icon: "brain.head.profile", label: "Training", color: .pink,   side: .left),
    PanelDef(id: "about",    icon: "info.circle",        label: "About",    color: .cyan,   side: .left),
    // Right column — top to bottom
    PanelDef(id: "stats",    icon: "chart.bar.xaxis",    label: "Stats",    color: .orange, side: .right),
    PanelDef(id: "settings", icon: "gearshape",          label: "Settings", color: .indigo, side: .right),
]

// MARK: - HUD Button

private struct HUDButton: View {
    let panel: PanelDef
    let isActive: Bool
    let onTap: () -> Void

    @State private var isHovered = false

    var body: some View {
        Button(action: onTap) {
            VStack(spacing: 5) {
                ZStack {
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(
                            isActive
                                ? panel.color.opacity(0.30)
                                : (isHovered ? panel.color.opacity(0.18) : panel.color.opacity(0.08))
                        )
                        .frame(width: 46, height: 46)

                    Image(systemName: panel.icon)
                        .font(.system(size: 18, weight: .medium))
                        .foregroundStyle(
                            isActive
                                ? panel.color
                                : (isHovered ? panel.color : panel.color.opacity(0.65))
                        )
                        .scaleEffect(isActive ? 1.1 : 1.0)
                        .animation(panelAnimation, value: isActive)
                }

                Text(panel.label)
                    .font(.system(size: 9.5, weight: .semibold))
                    .foregroundStyle(isActive ? panel.color : (isHovered ? .primary : .secondary))
            }
            .frame(width: 60, height: 64)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { isHovered = $0 }
        .animation(.easeInOut(duration: 0.13), value: isHovered)
        .help(panel.label)
    }
}

// MARK: - HUD Pill

private struct HUDPill: View {
    let panels: [PanelDef]
    @Binding var activePanels: Set<String>
    @Binding var isExpanded: Bool

    var body: some View {
        HStack(spacing: 2) {
            ForEach(panels) { panel in
                HUDButton(
                    panel: panel,
                    isActive: activePanels.contains(panel.id)
                ) {
                    withAnimation(panelAnimation) {
                        if activePanels.contains(panel.id) {
                            activePanels.remove(panel.id)
                        } else {
                            activePanels.insert(panel.id)
                        }
                    }
                }

                if panel.id != panels.last?.id {
                    Divider()
                        .frame(height: 28)
                        .opacity(0.25)
                        .padding(.horizontal, 2)
                }
            }

            Divider()
                .frame(height: 28)
                .opacity(0.25)
                .padding(.horizontal, 2)

            // Dismiss
            Button {
                withAnimation(panelAnimation) {
                    activePanels.removeAll()
                    isExpanded = false
                }
            } label: {
                Image(systemName: "chevron.down")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .frame(width: 36, height: 36)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help("Hide toolbar")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.ultraThinMaterial, in: Capsule())
        .overlay(Capsule().strokeBorder(.white.opacity(0.13), lineWidth: 1))
        .shadow(color: .black.opacity(0.40), radius: 20, y: 8)
    }
}

// MARK: - Reveal Button

private struct RevealButton: View {
    let action: () -> Void
    @State private var isHovered = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 6) {
                Image(systemName: "chevron.up")
                    .font(.system(size: 13, weight: .bold))
                Text("Menu")
                    .font(.system(size: 12, weight: .semibold))
            }
            .foregroundStyle(isHovered ? .primary : .secondary)
            .padding(.horizontal, 18)
            .padding(.vertical, 10)
            .background(.ultraThinMaterial, in: Capsule())
            .overlay(Capsule().strokeBorder(.white.opacity(isHovered ? 0.22 : 0.10), lineWidth: 1))
            .shadow(color: .black.opacity(0.30), radius: 10, y: 4)
            .opacity(isHovered ? 1.0 : 0.72)
            .scaleEffect(isHovered ? 1.04 : 1.0)
        }
        .buttonStyle(.plain)
        .onHover { isHovered = $0 }
        .animation(.easeInOut(duration: 0.15), value: isHovered)
        .help("Show toolbar")
    }
}

// MARK: - Panel Container

private struct PanelContainer<Content: View>: View {
    let panel: PanelDef
    let onClose: () -> Void
    let isTopInColumn: Bool
    let isBottomInColumn: Bool
    @ViewBuilder let content: () -> Content

    private var shape: some InsettableShape {
        let r: CGFloat = 16
        switch panel.side {
        case .left:
            return UnevenRoundedRectangle(
                topLeadingRadius: 0,
                bottomLeadingRadius: 0,
                bottomTrailingRadius: isBottomInColumn ? r : 0,
                topTrailingRadius: isTopInColumn ? r : 0
            )
        case .right:
            return UnevenRoundedRectangle(
                topLeadingRadius: isTopInColumn ? r : 0,
                bottomLeadingRadius: isBottomInColumn ? r : 0,
                bottomTrailingRadius: 0,
                topTrailingRadius: 0
            )
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            // Title bar
            HStack {
                HStack(spacing: 8) {
                    Circle()
                        .fill(panel.color)
                        .frame(width: 8, height: 8)
                        .shadow(color: panel.color.opacity(0.8), radius: 4)
                    Text(panel.label)
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(.primary)
                }
                Spacer()
                Button(action: onClose) {
                    Image(systemName: "xmark")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundStyle(.secondary)
                        .frame(width: 22, height: 22)
                        .background(.white.opacity(0.07), in: Circle())
                }
                .buttonStyle(.plain)
                .help("Close \(panel.label)")
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)

            Divider().opacity(0.3)

            content()
        }
        .background(Color(nsColor: .windowBackgroundColor).opacity(0.97))
        .overlay(shape.strokeBorder(.white.opacity(0.12), lineWidth: 1))
        .clipShape(shape)
        .shadow(color: .black.opacity(0.28), radius: 18, x: 0, y: 4)
    }
}

// MARK: - Side Column

/// Renders all visible panels for one side, stacked vertically with no gap so
/// shared interior edges look seamless.
private struct SideColumn: View {
    let side: PanelSide
    @Binding var activePanels: Set<String>

    private var visiblePanels: [PanelDef] {
        allPanels.filter { $0.side == side && activePanels.contains($0.id) }
    }

    var body: some View {
        let visible = visiblePanels
        if !visible.isEmpty {
            HStack(spacing: 0) {
                if side == .right { Spacer() }

                VStack(spacing: 0) {
                    ForEach(Array(visible.enumerated()), id: \.element.id) { index, panel in
                        let isTop    = index == 0
                        let isBottom = index == visible.count - 1

                        PanelContainer(
                            panel: panel,
                            onClose: {
                                withAnimation(panelAnimation) {
                                    _ = activePanels.remove(panel.id)
                                }
                            },
                            isTopInColumn: isTop,
                            isBottomInColumn: isBottom
                        ) {
                            if panel.id == "about" {
                                AboutView()
                            } else {
                                PlaceholderPanelView(
                                    title: panel.label,
                                    icon: panel.icon,
                                    color: panel.color,
                                    description: placeholderDescription(for: panel.id)
                                )
                            }
                        }
                        // Add a thin seam between stacked panels
                        if !isBottom {
                            Divider().opacity(0.25)
                        }
                    }
                }
                .frame(width: 260)

                if side == .left { Spacer() }
            }
            .transition(side.transition)
        }
    }

    private func placeholderDescription(for id: String) -> String {
        switch id {
        case "settings": return "Renderer settings will appear here."
        case "training": return "Neural network training controls will appear here."
        case "stats":    return "GPU timing and frame statistics will appear here."
        default:         return ""
        }
    }
}

// MARK: - ContentView

struct ContentView: View {

    @State private var renderer: MetalViewDelegate = {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("This device does not support Metal.")
        }
        return Renderer(device: device)
    }()

    @State private var isHUDExpanded: Bool = false
    @State private var activePanels: Set<String> = []

    var body: some View {
        ZStack {
            // Full-screen Metal render view
            MetalView(delegate: renderer)
                .ignoresSafeArea()

            // Left column (Training / About)
            SideColumn(side: .left, activePanels: $activePanels)

            // Right column (Stats / Settings)
            SideColumn(side: .right, activePanels: $activePanels)

            // Bottom HUD
            VStack {
                Spacer()

                if isHUDExpanded {
                    HUDPill(panels: allPanels, activePanels: $activePanels, isExpanded: $isHUDExpanded)
                        .transition(
                            .asymmetric(
                                insertion: .scale(scale: 0.80, anchor: .bottom)
                                    .combined(with: .opacity)
                                    .combined(with: .offset(y: 12)),
                                removal: .scale(scale: 0.80, anchor: .bottom)
                                    .combined(with: .opacity)
                                    .combined(with: .offset(y: 12))
                            )
                        )
                }

                if !isHUDExpanded {
                    RevealButton {
                        withAnimation(panelAnimation) {
                            isHUDExpanded = true
                        }
                    }
                    .transition(
                        .asymmetric(
                            insertion: .opacity.combined(with: .offset(y: 6)),
                            removal:   .opacity.combined(with: .offset(y: 6))
                        )
                    )
                }
            }
            .padding(.bottom, 18)
        }
    }
}

// MARK: - Placeholder Panel View

struct PlaceholderPanelView: View {
    let title: String
    let icon: String
    let color: Color
    let description: String

    var body: some View {
        VStack(spacing: 14) {
            Spacer()
            ZStack {
                Circle()
                    .fill(color.opacity(0.15))
                    .frame(width: 64, height: 64)
                Image(systemName: icon)
                    .font(.system(size: 28, weight: .light))
                    .foregroundStyle(color)
            }
            Text(title)
                .font(.title3)
                .fontWeight(.semibold)
            Text(description)
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 24)
            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
