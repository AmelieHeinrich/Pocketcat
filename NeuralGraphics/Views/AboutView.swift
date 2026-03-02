//
//  AboutView.swift
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 22/02/2026.
//

import SwiftUI

struct AboutView: View {
    private let version = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "1.0"
    private let build   = Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "1"

    var body: some View {
        VStack(spacing: 0) {
            // App icon
            Image(nsImage: NSApp.applicationIconImage)
                .resizable()
                .frame(width: 80, height: 80)
                .cornerRadius(16)
                .padding(.top, 24)
                .padding(.bottom, 12)

            // App name
            Text("Neural Graphics")
                .font(.title2)
                .fontWeight(.semibold)

            // Version + build
            Text("Version \(version) (build \(build))")
                .font(.caption)
                .foregroundStyle(.secondary)
            Text("February 2026")
                .font(.caption)
                .foregroundStyle(.secondary)
                .padding(.bottom, 16)

            Divider()
                .padding(.horizontal, 20)

            // Description
            VStack(alignment: .leading, spacing: 8) {
                Text("About")
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)

                Text("A Metal 4 renderer showcasing neural rendering techniques, including neural BRDFs and real-time inference on Apple Silicon GPUs.")
                    .font(.caption)
                    .foregroundStyle(.primary)
                    .multilineTextAlignment(.leading)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 14)

            Divider()
                .padding(.horizontal, 20)

            // Author
            VStack(spacing: 4) {
                Text("Author")
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundStyle(.secondary)
                Text("Amélie Heinrich")
                    .font(.caption)
            }
            .padding(.vertical, 14)

            Spacer()
        }
        .frame(maxWidth: .infinity)
        .frame(minHeight: 320)
    }
}

#Preview {
    AboutView()
}
