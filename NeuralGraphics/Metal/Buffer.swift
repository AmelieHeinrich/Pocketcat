//
//  Buffer.swift
//  Neural Graphics
//
//  Created by Amélie Heinrich on 25/02/2026.
//

import Metal

class Buffer {
    var buffer: MTLBuffer
    var size: Int
    
    init(size: Int) {
        self.buffer = RendererData.device.makeBuffer(length: size, options: .storageModeShared)!
        self.size = size
        
        RendererData.residencySet.addAllocation(self.buffer)
    }
    
    init(bytes: UnsafeRawPointer, size: Int) {
        self.buffer = RendererData.device.makeBuffer(bytes: bytes, length: size, options: .storageModeShared)!
        self.size = size
        
        RendererData.residencySet.addAllocation(self.buffer)
    }
    
    deinit {
        RendererData.residencySet.removeAllocation(self.buffer)
    }
    
    func setName(name: String) {
        self.buffer.label = name
    }
    
    func getAddress() -> MTLGPUAddress {
        return self.buffer.gpuAddress
    }
    
    func contents() -> UnsafeMutableRawPointer {
        return self.buffer.contents()
    }
    
    func write(bytes: UnsafeRawPointer, size: Int, offset: Int = 0) {
        memcpy(self.contents() + offset, bytes, size)
    }
}
