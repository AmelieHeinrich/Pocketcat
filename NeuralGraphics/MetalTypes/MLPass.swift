//
//  MLPass.swift
//  Neural Graphics
//
//  Created by Amélie Heinrich on 25/02/2026.
//

import Metal

class MLPass {
    var encoder: MTL4MachineLearningCommandEncoder
    
    init(name: String = "ML Pass", cmdBuffer: MTL4CommandBuffer) {
        encoder = cmdBuffer.makeMachineLearningCommandEncoder()!
        encoder.label = name
    }
    
    func end() {
        encoder.endEncoding()
    }
}
