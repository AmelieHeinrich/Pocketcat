//
//  NeuralNetwork.swift
//  Neural Graphics
//
//  Created by Amélie Heinrich on 27/02/2026.
//

import Metal

class NeuralNetwork {
    var pipeline: MTL4MachineLearningPipelineState
    var intermediateHeap: MTLHeap
    
    init(path: URL, name: String = "Neural Network") {
        let library = try! RendererData.device.makeLibrary(URL: path)
        
        let functionDescriptor = MTL4LibraryFunctionDescriptor()
        functionDescriptor.library = library
        functionDescriptor.name = "main"
        
        let descriptor = MTL4MachineLearningPipelineDescriptor()
        descriptor.label = name
        descriptor.machineLearningFunctionDescriptor = functionDescriptor
        
        self.pipeline = try! RendererData.compiler.makeMachineLearningPipelineState(descriptor: descriptor)
        
        let heapSize = self.pipeline.intermediatesHeapSize
        
        let heapDesc = MTLHeapDescriptor()
        heapDesc.size = heapSize
        heapDesc.hazardTrackingMode = .default
        heapDesc.resourceOptions = .storageModeShared
        heapDesc.type = .placement
        
        self.intermediateHeap = RendererData.device.makeHeap(descriptor: heapDesc)!
        
        let reflection = self.pipeline.reflection!
        reflection.bindings.forEach() { binding in
            if binding.type == .tensor {
                let tensorBinding = binding as! any MTLTensorBinding as MTLTensorBinding
                print(tensorBinding.dimensions!)
            }
        }
    }
}
