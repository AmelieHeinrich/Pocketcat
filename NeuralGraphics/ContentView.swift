//
//  ContentView.swift
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 22/02/2026.
//

import SwiftUI

struct ContentView: View {
    var body: some View {
        HSplitView {
            // SwiftUI panel
            VStack {
                Spacer()
            }
            .frame(minWidth: 200, maxWidth: 300)

            // Metal render view
            MetalView()
                .frame(minWidth: 400, maxWidth: .infinity)
        }
    }
}
