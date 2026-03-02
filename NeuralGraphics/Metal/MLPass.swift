//
//  MLPass.swift
//  Neural Graphics
//
//  Created by Amélie Heinrich on 25/02/2026.
//

import Metal

class MLPass {
    var encoder: MTL4MachineLearningCommandEncoder
    var currentPipeline: NeuralNetwork!
    
    init(name: String = "ML Pass", cmdBuffer: MTL4CommandBuffer) {
        encoder = cmdBuffer.makeMachineLearningCommandEncoder()!
        encoder.label = name
    }
    
    func end() {
        encoder.endEncoding()
    }
    
    func setNeuralNetwork(nn: NeuralNetwork) {
        self.currentPipeline = nn
        
        encoder.setArgumentTable(RendererData.mlTable)
        encoder.setPipelineState(nn.pipeline)
    }
    
    func setTensors(tensors: [Tensor]) {
        RendererData.mlTable.setResource(tensors[0].getAddress(), bufferIndex: 0)
        RendererData.mlTable.setResource(tensors[1].getAddress(), bufferIndex: 1)
    }
    
    func infer() {
        encoder.dispatchNetwork(intermediatesHeap: self.currentPipeline.intermediateHeap)
    }
}
