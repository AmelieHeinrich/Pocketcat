//
//  RendererData.swift
//  Pocketcat
//
//  Created by Amélie Heinrich on 25/02/2026.
//

import Metal
import AppKit

struct RendererData {
    static var device: MTLDevice!
    static var cmdQueue: MTL4CommandQueue!
    static var residencySet: MTLResidencySet!
    static var compiler: MTL4Compiler!
    private static let residencyLock = NSLock()

    static var mtl3commandQueue: MTLCommandQueue!
    static var mtl3commandBuffer: MTLCommandBuffer!
    
    static func getPixelFormat() -> MTLPixelFormat {
        let supportsHDR = (NSScreen.main?.maximumPotentialExtendedDynamicRangeColorComponentValue ?? 1.0) > 1.0
        return supportsHDR ? .rgba16Float : .bgra8Unorm
    }

    static func addResidentAllocation(_ allocation: some MTLAllocation) {
        residencyLock.withLock { residencySet.addAllocation(allocation) }
    }

    static func removeResidentAllocation(_ allocation: some MTLAllocation) {
        residencyLock.withLock { residencySet.removeAllocation(allocation) }
    }

    static func commitResidency() {
        residencyLock.withLock { residencySet.commit() }
    }
    static var library: MTLLibrary!
    static var gpuTimeline: GPUTimeline!

    static var vertexTable: MTL4ArgumentTable!
    static var fragmentTable: MTL4ArgumentTable!
    static var meshTable: MTL4ArgumentTable!
    static var objectTable: MTL4ArgumentTable!
    static var computeTable: MTL4ArgumentTable!
    static var tileTable: MTL4ArgumentTable!
    static var mlTable: MTL4ArgumentTable!

    // Counter heap for per-encoder GPU timing (Metal4)
    // nil if device doesn't support timestamp counters
    static var counterHeap: (any MTL4CounterHeap)?
    static let counterHeapSlotsPerFrame: Int = 64  // 2 × max_encoders per frame

    // Render-thread-only state — reset at the start of each frame's recording window
    static var counterOffset: Int = 0
    static var counterEntries: [(name: String, startSlot: Int, endSlot: Int)] = []

    static func initialize(
        device: MTLDevice,
        cmdQueue: MTL4CommandQueue,
        residencySet: MTLResidencySet,
        compiler: MTL4Compiler
    ) {
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
        argumentTableDescriptor.label = "Vertex Argument Table"

        self.vertexTable = try! self.device.makeArgumentTable(descriptor: argumentTableDescriptor)

        argumentTableDescriptor.label = "Fragment Argument Table"
        self.fragmentTable = try! self.device.makeArgumentTable(descriptor: argumentTableDescriptor)

        argumentTableDescriptor.label = "Mesh Argument Table"
        self.meshTable = try! self.device.makeArgumentTable(descriptor: argumentTableDescriptor)

        argumentTableDescriptor.label = "Object Argument Table"
        self.objectTable = try! self.device.makeArgumentTable(descriptor: argumentTableDescriptor)

        argumentTableDescriptor.label = "Compute Argument Table"
        self.computeTable = try! self.device.makeArgumentTable(descriptor: argumentTableDescriptor)

        argumentTableDescriptor.label = "Tile Argument Table"
        self.tileTable = try! self.device.makeArgumentTable(descriptor: argumentTableDescriptor)

        argumentTableDescriptor.label = "ML Argument Table"
        self.mlTable = try! self.device.makeArgumentTable(descriptor: argumentTableDescriptor)

        self.mtl3commandQueue = device.makeCommandQueue()
        self.mtl3commandBuffer = mtl3commandQueue.makeCommandBuffer()

        // Counter heap for GPU pass timing
        let heapDesc = MTL4CounterHeapDescriptor()
        heapDesc.type = .timestamp
        heapDesc.count = counterHeapSlotsPerFrame * 3
        Self.counterHeap = try? device.makeCounterHeap(descriptor: heapDesc)
    }

    static func waitIdle() {
        let done = RendererData.device.makeSharedEvent()!
        RendererData.cmdQueue.signalEvent(done, value: 1)
        done.wait(untilSignaledValue: 1, timeoutMS: 10_000)
    }
}
