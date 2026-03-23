//
//  WorldView.swift
//  Pocketcat
//
//  Created by Amélie Heinrich on 2026.
//

import SwiftUI

// MARK: - Sun Arcball Gizmo

private struct SunArcballGizmo: View {
    @Binding var azimuth: Float    // degrees
    @Binding var elevation: Float  // degrees

    private let size: CGFloat = 130

    var body: some View {
        Canvas { ctx, canvasSize in
            let cx = canvasSize.width / 2
            let cy = canvasSize.height / 2
            let r = min(cx, cy) - 12

            // Background disc
            ctx.fill(
                Path(ellipseIn: CGRect(x: cx - r, y: cy - r, width: r * 2, height: r * 2)),
                with: .color(.white.opacity(0.05)))

            // Outer ring
            var outerPath = Path()
            outerPath.addEllipse(in: CGRect(x: cx - r, y: cy - r, width: r * 2, height: r * 2))
            ctx.stroke(outerPath, with: .color(.white.opacity(0.2)), lineWidth: 1)

            // Elevation rings at 30° and 60°
            for deg in [30.0, 60.0] {
                let frac = CGFloat(deg / 90.0)
                let ringR = r * (1.0 - frac)
                var ringPath = Path()
                ringPath.addEllipse(in: CGRect(x: cx - ringR, y: cy - ringR, width: ringR * 2, height: ringR * 2))
                ctx.stroke(ringPath, with: .color(.white.opacity(0.10)), lineWidth: 0.5)
            }

            // Compass labels
            let labels: [(String, CGFloat, CGFloat)] = [
                ("N", cx, cy - r - 9),
                ("S", cx, cy + r + 9),
                ("E", cx + r + 9, cy),
                ("W", cx - r - 9, cy)
            ]
            for (label, lx, ly) in labels {
                ctx.draw(
                    Text(label).font(.system(size: 8, weight: .semibold)).foregroundStyle(.secondary),
                    at: CGPoint(x: lx, y: ly))
            }

            // Horizon line (below-horizon region tinted dark)
            if elevation < 0 {
                let nightPath = Path(ellipseIn: CGRect(x: cx - r, y: cy - r, width: r * 2, height: r * 2))
                ctx.fill(nightPath, with: .color(.black.opacity(0.35)))
            }

            // Sun dot position: azimuth rotates around ring, elevation shrinks radius
            let az = CGFloat(azimuth) * .pi / 180.0
            let el = CGFloat(elevation) / 90.0  // 0=horizon, 1=zenith
            let dotR = r * (1.0 - max(0, el))
            let dotX = cx + dotR * sin(az)
            let dotY = cy - dotR * cos(az)

            // Glow
            ctx.fill(
                Path(ellipseIn: CGRect(x: dotX - 9, y: dotY - 9, width: 18, height: 18)),
                with: .color(.yellow.opacity(0.25)))

            // Sun dot
            ctx.fill(
                Path(ellipseIn: CGRect(x: dotX - 5, y: dotY - 5, width: 10, height: 10)),
                with: elevation >= 0 ? .color(.yellow) : .color(.gray.opacity(0.5)))
        }
        .frame(width: size, height: size)
        .gesture(
            DragGesture(minimumDistance: 0)
                .onChanged { value in
                    let cx = size / 2
                    let cy = size / 2
                    let r = min(cx, cy) - 12

                    let dx = value.location.x - cx
                    let dy = value.location.y - cy
                    let dist = sqrt(dx * dx + dy * dy)

                    // Azimuth from angle
                    let angle = atan2(dx, -dy) * 180.0 / .pi
                    azimuth = Float((angle + 360).truncatingRemainder(dividingBy: 360))

                    // Elevation from distance: center=zenith(90°), rim=horizon(0°)
                    let frac = min(dist / r, 1.0)
                    elevation = Float((1.0 - frac) * 90.0)
                }
        )
        .help("Drag to set sun direction")
    }
}

// MARK: - Point Light Row

private struct PointLightRow: View {
    @Binding var light: PointLightEntry
    let onRemove: () -> Void

    var body: some View {
        VStack(spacing: 6) {
            HStack {
                ColorPicker("", selection: $light.color)
                    .labelsHidden()
                    .frame(width: 28)
                Text("Light")
                    .font(.system(size: 11, weight: .semibold))
                Spacer()
                Button(action: onRemove) {
                    Image(systemName: "minus.circle.fill")
                        .foregroundStyle(.red.opacity(0.8))
                        .font(.system(size: 14))
                }
                .buttonStyle(.plain)
            }
            HStack(spacing: 4) {
                Text("X").font(.system(size: 9)).foregroundStyle(.secondary).frame(width: 10)
                TextField("0", value: $light.position.x, format: .number)
                    .textFieldStyle(.plain)
                    .font(.system(size: 10, design: .monospaced))
                    .frame(maxWidth: .infinity)
                    .padding(.horizontal, 4).padding(.vertical, 2)
                    .background(.white.opacity(0.05), in: RoundedRectangle(cornerRadius: 4))
                Text("Y").font(.system(size: 9)).foregroundStyle(.secondary).frame(width: 10)
                TextField("0", value: $light.position.y, format: .number)
                    .textFieldStyle(.plain)
                    .font(.system(size: 10, design: .monospaced))
                    .frame(maxWidth: .infinity)
                    .padding(.horizontal, 4).padding(.vertical, 2)
                    .background(.white.opacity(0.05), in: RoundedRectangle(cornerRadius: 4))
                Text("Z").font(.system(size: 9)).foregroundStyle(.secondary).frame(width: 10)
                TextField("0", value: $light.position.z, format: .number)
                    .textFieldStyle(.plain)
                    .font(.system(size: 10, design: .monospaced))
                    .frame(maxWidth: .infinity)
                    .padding(.horizontal, 4).padding(.vertical, 2)
                    .background(.white.opacity(0.05), in: RoundedRectangle(cornerRadius: 4))
            }
            HStack {
                Text("Radius").font(.system(size: 10)).foregroundStyle(.secondary)
                Slider(value: $light.radius, in: 0.1...50.0)
                Text(String(format: "%.1f", light.radius)).font(.system(size: 10, design: .monospaced)).foregroundStyle(.secondary).frame(width: 28)
            }
            HStack {
                Text("Intensity").font(.system(size: 10)).foregroundStyle(.secondary)
                Slider(value: $light.intensity, in: 0.0...20.0)
                Text(String(format: "%.1f", light.intensity)).font(.system(size: 10, design: .monospaced)).foregroundStyle(.secondary).frame(width: 28)
            }
        }
        .padding(8)
        .background(.white.opacity(0.04), in: RoundedRectangle(cornerRadius: 8))
    }
}

// MARK: - WorldView

struct WorldView: View {
    @ObservedObject var lightState: LightState

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                sunSection
                Divider().opacity(0.2).padding(.vertical, 8)
                pointLightsSection
            }
            .padding(14)
        }
    }

    // MARK: Sun Section

    private var sunSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("Sun Light", systemImage: "sun.max.fill")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.yellow)

            Picker("Mode", selection: $lightState.sunMode) {
                ForEach(SunMode.allCases, id: \.self) { mode in
                    Text(mode.rawValue).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()

            if lightState.sunMode == .manual {
                HStack {
                    Spacer()
                    SunArcballGizmo(azimuth: $lightState.sunAzimuth, elevation: $lightState.sunElevation)
                    Spacer()
                }

                LabeledSlider(label: "Azimuth", value: $lightState.sunAzimuth, range: 0...360,
                              format: "%.0f°")
                LabeledSlider(label: "Elevation", value: $lightState.sunElevation, range: -10...90,
                              format: "%.0f°")
            } else {
                HStack(spacing: 6) {
                    Image(systemName: "clock")
                        .foregroundStyle(.secondary)
                        .font(.system(size: 11))
                    Text("Synced to system clock")
                        .font(.system(size: 10))
                        .foregroundStyle(.secondary)
                }
            }

            HStack(spacing: 8) {
                Text("Color").font(.system(size: 11)).foregroundStyle(.secondary)
                ColorPicker("", selection: $lightState.sunColor)
                    .labelsHidden()
                    .frame(width: 32)
                Spacer()
            }

            if lightState.sunMode == .manual {
                LabeledSlider(label: "Intensity", value: $lightState.sunIntensity, range: 0.0...15.0,
                              format: "%.1f")
            }

            LabeledSlider(label: "RT Radius", value: $lightState.sunRadius, range: 0.0...0.5,
                          format: "%.3f")
        }
    }

    // MARK: Point Lights Section

    private var pointLightsSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Label("Point Lights", systemImage: "lightbulb.fill")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.orange)
                Spacer()
                Button {
                    lightState.pointLights.append(PointLightEntry())
                } label: {
                    Image(systemName: "plus.circle.fill")
                        .foregroundStyle(.orange.opacity(0.8))
                        .font(.system(size: 16))
                }
                .buttonStyle(.plain)
                .help("Add point light")
            }

            if lightState.pointLights.isEmpty {
                Text("No point lights")
                    .font(.system(size: 11))
                    .foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8)
            } else {
                ForEach($lightState.pointLights) { $light in
                    PointLightRow(light: $light) {
                        lightState.pointLights.removeAll { $0.id == light.id }
                    }
                }
            }
        }
    }
}

// MARK: - Labeled Slider

private struct LabeledSlider: View {
    let label: String
    @Binding var value: Float
    let range: ClosedRange<Float>
    let format: String

    var body: some View {
        HStack(spacing: 6) {
            Text(label)
                .font(.system(size: 10))
                .foregroundStyle(.secondary)
                .frame(width: 56, alignment: .leading)
            Slider(value: $value, in: range)
            Text(String(format: format, value))
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(.secondary)
                .frame(width: 38, alignment: .trailing)
        }
    }
}
