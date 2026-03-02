//
//  ComputePass.swift
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 02/03/2026.
//

import Metal

class ComputePass {
    var encoder: MTL4ComputeCommandEncoder
    
    init(label: String, cmdBuffer: MTL4CommandBuffer) {
        self.encoder = cmdBuffer.makeComputeCommandEncoder()!
        self.encoder.label = label
    }
    
    func end() {
        self.encoder.endEncoding()
    }
}
