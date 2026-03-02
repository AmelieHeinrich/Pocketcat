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
    var allocated: Bool = false

    init(size: Int, makeResidentNow: Bool = true) {
        self.buffer = RendererData.device.makeBuffer(length: size, options: .storageModeShared)!
        self.size = size

        if makeResidentNow {
            RendererData.residencySet.addAllocation(self.buffer)
            self.allocated = true
        }
    }

    init(bytes: UnsafeRawPointer, size: Int, makeResidentNow: Bool = true) {
        self.buffer = RendererData.device.makeBuffer(bytes: bytes, length: size, options: .storageModeShared)!
        self.size = size

        if makeResidentNow {
            RendererData.residencySet.addAllocation(self.buffer)
            self.allocated = true
        }
    }

    deinit {
        if self.allocated {
            RendererData.residencySet.removeAllocation(self.buffer)
        }
    }

    func makeResident() {
        guard !allocated else { return }
        RendererData.residencySet.addAllocation(self.buffer)
        allocated = true
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
