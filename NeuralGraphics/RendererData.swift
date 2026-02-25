//
//  RendererData.swift
//  Neural Graphics
//
//  Created by Amélie Heinrich on 25/02/2026.
//

import Metal

struct RendererData {
    static var device: MTLDevice!
    static var cmdQueue: MTL4CommandQueue!
    static var residencySet: MTLResidencySet!
    static var compiler: MTL4Compiler!
    static var library: MTLLibrary!
    static var gpuTimeline: GPUTimeline!
    
    static func initialize(device: MTLDevice,
                           cmdQueue: MTL4CommandQueue,
                           residencySet: MTLResidencySet,
                           compiler: MTL4Compiler) {
        self.device = device
        self.cmdQueue = cmdQueue
        self.residencySet = residencySet
        self.compiler = compiler
        self.library = self.device.makeDefaultLibrary()!
        self.gpuTimeline = GPUTimeline()
    }
}
