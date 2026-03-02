//
//  RenderTimeline.swift
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 02/03/2026.
//

// RenderTimeline holds *weak* handles to passes — it does not own them.
// FrameManager owns the passes; the timeline is just the ordered execution plan.
// This lets you hot-swap timelines (e.g. DesktopTimeline vs MobileTimeline)
// without moving pass ownership.

private final class WeakPass {
    weak var value: Pass?
    init(_ pass: Pass) { self.value = pass }
}

class RenderTimeline {
    private var passes: [WeakPass] = []

    func addPass(_ pass: Pass) {
        passes.append(WeakPass(pass))
    }

    func execute(context: FrameContext) {
        for box in passes {
            box.value?.render(context: context)
        }
    }
}
