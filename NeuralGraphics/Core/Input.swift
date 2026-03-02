//
//  Input.swift
//  Neural Graphics
//
//  Created by Amélie Heinrich on 28/02/2026.
//

import AppKit

class Input {
    static let shared = Input()

    private var keysDown: Set<UInt16> = []
    private var keysPressed: Set<UInt16> = []  // set only for one frame
    private var keysReleased: Set<UInt16> = [] // set only for one frame

    private(set) var mouseDelta: SIMD2<Float> = .zero
    private(set) var mousePosition: SIMD2<Float> = .zero

    private(set) var leftMouseDown: Bool = false
    private(set) var rightMouseDown: Bool = false

    // Call once per frame before polling input
    func beginFrame() {
        keysPressed.removeAll(keepingCapacity: true)
        keysReleased.removeAll(keepingCapacity: true)
        mouseDelta = .zero
    }

    // MARK: - Keyboard

    func keyDown(event: NSEvent) {
        guard !event.isARepeat else { return }
        keysDown.insert(event.keyCode)
        keysPressed.insert(event.keyCode)
    }

    func keyUp(event: NSEvent) {
        keysDown.remove(event.keyCode)
        keysReleased.insert(event.keyCode)
    }

    func isKeyDown(_ key: KeyCode) -> Bool {
        keysDown.contains(key.rawValue)
    }

    func isKeyPressed(_ key: KeyCode) -> Bool {
        keysPressed.contains(key.rawValue)
    }

    func isKeyReleased(_ key: KeyCode) -> Bool {
        keysReleased.contains(key.rawValue)
    }

    // MARK: - Mouse

    func mouseMoved(event: NSEvent, in view: NSView) {
        let location = view.convert(event.locationInWindow, from: nil)
        mousePosition = SIMD2<Float>(Float(location.x), Float(location.y))
        // Accumulate raw deltas — these are valid for both mouseMoved and drag events
        mouseDelta.x += Float(event.deltaX)
        mouseDelta.y += Float(event.deltaY)
    }

    func mouseDown(event: NSEvent) {
        if event.type == .leftMouseDown  { leftMouseDown = true }
        if event.type == .rightMouseDown {
            rightMouseDown = true
            CGDisplayHideCursor(CGMainDisplayID())
        }
    }

    func mouseUp(event: NSEvent) {
        if event.type == .leftMouseUp  { leftMouseDown = false }
        if event.type == .rightMouseUp {
            rightMouseDown = false
            CGDisplayShowCursor(CGMainDisplayID())
        }
    }
}

// MARK: - Key codes (US layout, macOS virtual key codes)

enum KeyCode: UInt16 {
    case a = 0, s = 1, d = 2, f = 3
    case h = 4, g = 5, z = 6, x = 7
    case c = 8, v = 9, b = 11, q = 12
    case w = 13, e = 14, r = 15, y = 16
    case t = 17, one = 18, two = 19, three = 20
    case four = 21, six = 22, five = 23, equal = 24
    case nine = 25, seven = 26, minus = 27, eight = 28
    case zero = 29, rightBracket = 30, o = 31, u = 32
    case leftBracket = 33, i = 34, p = 35
    case `return` = 36, l = 37, j = 38
    case quote = 39, k = 40, semicolon = 41
    case backslash = 42, comma = 43, slash = 44
    case n = 45, m = 46, period = 47
    case tab = 48, space = 49, grave = 50
    case delete = 51, escape = 53
    case command = 55, shift = 56, capsLock = 57
    case option = 58, control = 59
    case rightShift = 60, rightOption = 61, rightControl = 62
    case f17 = 64, keypadDecimal = 65
    case keypadMultiply = 67, keypadPlus = 69
    case keypadClear = 71, keypadDivide = 75
    case keypadEnter = 76, keypadMinus = 78
    case f18 = 79, f19 = 80, keypadEqual = 81
    case keypad0 = 82, keypad1 = 83, keypad2 = 84
    case keypad3 = 85, keypad4 = 86, keypad5 = 87
    case keypad6 = 88, keypad7 = 89, keypad8 = 91
    case keypad9 = 92
    case f5 = 96, f6 = 97, f7 = 98, f3 = 99
    case f8 = 100, f9 = 101, f11 = 103
    case f13 = 105, f16 = 106, f14 = 107
    case f10 = 109, f12 = 111, f15 = 113
    case help = 114, home = 115, pageUp = 116
    case forwardDelete = 117, f4 = 118, end = 119
    case f2 = 120, pageDown = 121, f1 = 122
    case leftArrow = 123, rightArrow = 124
    case downArrow = 125, upArrow = 126
}
