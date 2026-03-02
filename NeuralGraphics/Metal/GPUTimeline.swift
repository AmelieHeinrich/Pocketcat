//
//  GPUTimeline.swift
//  Neural Graphics
//
//  Created by Amélie Heinrich on 25/02/2026.
//

import Metal

class GPUTimeline {
    var currentValue: UInt64
    var event: MTLSharedEvent
    var fence: MTLFence
    
    init() {
        self.currentValue = 0
        self.event = RendererData.device.makeSharedEvent()!
        self.fence = RendererData.device.makeFence()!
    }
    
    func signal() -> UInt64 {
        self.currentValue += 1
        RendererData.cmdQueue.signalEvent(self.event, value: self.currentValue)
        return self.currentValue
    }
    
    func wait(value: UInt64) {
        event.wait(untilSignaledValue: value, timeoutMS: 8)
    }
}
