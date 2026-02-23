//
//  ContentView.swift
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 22/02/2026.
//

import SwiftUI

struct ContentView: View {
    @State private var renderer: MetalViewDelegate = {
            guard let metalDevice = MTLCreateSystemDefaultDevice() else {
                fatalError("This sample requires a device that supports Metal")
            }

        return Renderer(device: metalDevice)
    }()
    
    var body: some View {
        HSplitView {
            // SwiftUI panel
            VStack {
                Spacer()
            }
            .frame(minWidth: 200, maxWidth: 300)

            // Metal render view
            MetalView(delegate: renderer)
                .frame(minWidth: 400, maxWidth: .infinity)
        }
    }
}
