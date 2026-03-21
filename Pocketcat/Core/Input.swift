//
//  Input.swift
//  Pocketcat
//
//  Created by Amélie Heinrich on 28/02/2026.
//

import AppKit

class Input {
    static let shared = Input()

    private var keysDown: Set<UInt16> = []
    private var keysPressed: Set<UInt16> = []  // set only for one frame
    private var keysReleased: Set<UInt16> = []  // set only for one frame

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
        if event.type == .leftMouseDown { leftMouseDown = true }
        if event.type == .rightMouseDown {
            rightMouseDown = true
            CGDisplayHideCursor(CGMainDisplayID())
        }
    }

    func mouseUp(event: NSEvent) {
        if event.type == .leftMouseUp { leftMouseDown = false }
        if event.type == .rightMouseUp {
            rightMouseDown = false
            CGDisplayShowCursor(CGMainDisplayID())
        }
    }

    func releaseAll() {
        keysDown.removeAll(keepingCapacity: true)
        keysPressed.removeAll(keepingCapacity: true)
        keysReleased.removeAll(keepingCapacity: true)
        leftMouseDown = false
        if rightMouseDown {
            rightMouseDown = false
            CGDisplayShowCursor(CGMainDisplayID())
        }
    }
}

// MARK: - Key codes (US layout, macOS virtual key codes)

enum KeyCode: UInt16 {
    case a = 0
    case s = 1
    case d = 2
    case f = 3
    case h = 4
    case g = 5
    case z = 6
    case x = 7
    case c = 8
    case v = 9
    case b = 11
    case q = 12
    case w = 13
    case e = 14
    case r = 15
    case y = 16
    case t = 17
    case one = 18
    case two = 19
    case three = 20
    case four = 21
    case six = 22
    case five = 23
    case equal = 24
    case nine = 25
    case seven = 26
    case minus = 27
    case eight = 28
    case zero = 29
    case rightBracket = 30
    case o = 31
    case u = 32
    case leftBracket = 33
    case i = 34
    case p = 35
    case `return` = 36
    case l = 37
    case j = 38
    case quote = 39
    case k = 40
    case semicolon = 41
    case backslash = 42
    case comma = 43
    case slash = 44
    case n = 45
    case m = 46
    case period = 47
    case tab = 48
    case space = 49
    case grave = 50
    case delete = 51
    case escape = 53
    case command = 55
    case shift = 56
    case capsLock = 57
    case option = 58
    case control = 59
    case rightShift = 60
    case rightOption = 61
    case rightControl = 62
    case f17 = 64
    case keypadDecimal = 65
    case keypadMultiply = 67
    case keypadPlus = 69
    case keypadClear = 71
    case keypadDivide = 75
    case keypadEnter = 76
    case keypadMinus = 78
    case f18 = 79
    case f19 = 80
    case keypadEqual = 81
    case keypad0 = 82
    case keypad1 = 83
    case keypad2 = 84
    case keypad3 = 85
    case keypad4 = 86
    case keypad5 = 87
    case keypad6 = 88
    case keypad7 = 89
    case keypad8 = 91
    case keypad9 = 92
    case f5 = 96
    case f6 = 97
    case f7 = 98
    case f3 = 99
    case f8 = 100
    case f9 = 101
    case f11 = 103
    case f13 = 105
    case f16 = 106
    case f14 = 107
    case f10 = 109
    case f12 = 111
    case f15 = 113
    case help = 114
    case home = 115
    case pageUp = 116
    case forwardDelete = 117
    case f4 = 118
    case end = 119
    case f2 = 120
    case pageDown = 121
    case f1 = 122
    case leftArrow = 123
    case rightArrow = 124
    case downArrow = 125
    case upArrow = 126
}
