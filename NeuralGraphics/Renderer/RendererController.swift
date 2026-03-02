//
//  RendererController.swift
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 02/03/2026.
//

// RendererController is the strategy for how a frame is driven.
// Swap it out to change the entire rendering behaviour — camera system,
// what data gets populated, what side effects happen (e.g. texture capture
// for NNAO training). The timeline itself stays the same; only the driver changes.

class RendererController {
    func resize(width: Int, height: Int) {}
    func render(timeline: RenderTimeline, context: inout FrameContext) {}
}
