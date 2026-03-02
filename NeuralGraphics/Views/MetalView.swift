//
//  MetalView.swift
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 22/02/2026.
//

import SwiftUI
import MetalKit

// MTKView subclass that forwards keyboard and mouse events to Input.shared
class InputMTKView: MTKView {
    private var rightMouseMonitor: Any?
    private var rightMouseUpMonitor: Any?

    override var acceptsFirstResponder: Bool { true }

    override func keyDown(with event: NSEvent) {
        Input.shared.keyDown(event: event)
    }

    override func keyUp(with event: NSEvent) {
        Input.shared.keyUp(event: event)
    }

    override func mouseMoved(with event: NSEvent) {
        Input.shared.mouseMoved(event: event, in: self)
    }

    override func mouseDragged(with event: NSEvent) {
        Input.shared.mouseMoved(event: event, in: self)
    }

    override func rightMouseDragged(with event: NSEvent) {
        Input.shared.mouseMoved(event: event, in: self)
    }

    override func mouseDown(with event: NSEvent) {
        Input.shared.mouseDown(event: event)
    }

    override func mouseUp(with event: NSEvent) {
        Input.shared.mouseUp(event: event)
    }

    override func rightMouseDown(with event: NSEvent) {
        Input.shared.mouseDown(event: event)
    }

    override func rightMouseUp(with event: NSEvent) {
        Input.shared.mouseUp(event: event)
    }

    override func menu(for event: NSEvent) -> NSMenu? {
        return nil
    }

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        trackingAreas.forEach { removeTrackingArea($0) }
        addTrackingArea(NSTrackingArea(
            rect: bounds,
            options: [.activeInKeyWindow, .mouseMoved, .inVisibleRect],
            owner: self
        ))
    }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        window?.makeFirstResponder(self)

        // SwiftUI can swallow rightMouseDragged events before they reach the view,
        // so we install a global monitor to capture them unconditionally.
        rightMouseMonitor = NSEvent.addLocalMonitorForEvents(matching: [.rightMouseDragged, .rightMouseDown, .rightMouseUp]) { [weak self] event in
            guard let self = self else { return event }
            switch event.type {
            case .rightMouseDragged:
                Input.shared.mouseMoved(event: event, in: self)
            case .rightMouseDown:
                Input.shared.mouseDown(event: event)
            case .rightMouseUp:
                Input.shared.mouseUp(event: event)
            default:
                break
            }
            return event
        }
    }

    deinit {
        if let monitor = rightMouseMonitor {
            NSEvent.removeMonitor(monitor)
        }
    }
}

struct MetalView: NSViewRepresentable {
    public typealias NSViewType = InputMTKView
    public var delegate: MetalViewDelegate?

    public init(delegate: MetalViewDelegate) {
        self.delegate = delegate
    }

    public func makeNSView(context: Context) -> InputMTKView {
        return InputMTKView()
    }

    public func updateNSView(_ view: InputMTKView, context: Context) {
        delegate?.configure(view)
    }
}
