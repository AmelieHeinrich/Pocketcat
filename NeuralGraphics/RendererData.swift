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
    
    static var vertexTable: MTL4ArgumentTable!
    static var fragmentTable: MTL4ArgumentTable!
    static var meshTable: MTL4ArgumentTable!
    static var objectTable: MTL4ArgumentTable!
    static var computeTable: MTL4ArgumentTable!
    static var tileTable: MTL4ArgumentTable!
    
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
        
        let argumentTableDescriptor = MTL4ArgumentTableDescriptor()
        argumentTableDescriptor.maxBufferBindCount = 16
        argumentTableDescriptor.maxTextureBindCount = 16
        argumentTableDescriptor.maxSamplerStateBindCount = 16
        
        self.vertexTable = try! self.device.makeArgumentTable(descriptor: argumentTableDescriptor)
        self.fragmentTable = try! self.device.makeArgumentTable(descriptor: argumentTableDescriptor)
        self.meshTable = try! self.device.makeArgumentTable(descriptor: argumentTableDescriptor)
        self.objectTable = try! self.device.makeArgumentTable(descriptor: argumentTableDescriptor)
        self.computeTable = try! self.device.makeArgumentTable(descriptor: argumentTableDescriptor)
        self.tileTable = try! self.device.makeArgumentTable(descriptor: argumentTableDescriptor)
    }
}
