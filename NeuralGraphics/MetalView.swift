//
//  MetalView.swift
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 22/02/2026.
//

import SwiftUI
import MetalKit

struct MetalView : NSViewRepresentable {
    public typealias NSViewType = MTKView
    public var delegate: MetalViewDelegate?

    public init(delegate: MetalViewDelegate) {
        self.delegate = delegate
    }

    public func makeNSView(context: Context) -> MTKView {
        return MTKView()
    }

    public func updateNSView(_ view: MTKView, context: Context) {
        delegate?.configure(view)
    }
}
