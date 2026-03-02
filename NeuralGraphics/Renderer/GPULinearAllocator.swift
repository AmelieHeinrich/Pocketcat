//
//  GPULinearAllocator.swift
//  Neural Graphics
//
//  Created by Amélie Heinrich on 28/02/2026.
//

class GPULinearAllocator {
    var buffer: Buffer
    var currOffset: Int
    var size: Int
    
    init(size: Int) {
        self.currOffset = 0
        self.size = size
        self.buffer = Buffer(size: size)
        self.buffer.setName(name: "GPU Linear Allocator Buffer")
    }
    
    func allocate(size: Int) -> Int {
        assert(self.currOffset + size <= self.size)
        
        let offset = self.currOffset
        self.currOffset += size
        
        return offset
    }
    
    func writeData(data: UnsafeRawPointer, offset: Int, size: Int) {
        buffer.write(bytes: data, size: size, offset: offset)
    }
    
    func reset() {
        self.currOffset = 0
    }
}
