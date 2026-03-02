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
    var tensorBindings: [MTLTensorBinding]
    
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
        heapDesc.type = .placement
        
        self.intermediateHeap = RendererData.device.makeHeap(descriptor: heapDesc)!
        
        self.tensorBindings = []
        let reflection = self.pipeline.reflection
        if reflection != nil {
            reflection?.bindings.forEach() { binding in
                if binding.type == .tensor {
                    self.tensorBindings.append(binding as! any MTLTensorBinding as MTLTensorBinding)
                }
            }
        }
    }
    
    func createTensors() -> [Tensor] {
        var tensors: [Tensor] = []
        
        for binding in tensorBindings {
            let dimensions = binding.dimensions
            let dataType = binding.tensorDataType
            
            if dimensions == nil {
                print("Dynamic dimensions is not supported")
                return []
            }
            
            let descriptor = MTLTensorDescriptor()
            descriptor.dimensions = dimensions!
            descriptor.dataType = dataType
            descriptor.usage = [.machineLearning, .compute, .render]
            
            let newTensor = Tensor(descriptor: descriptor)
            newTensor.setName(name: "Tensor: " + binding.name)
            
            tensors.append(newTensor)
        }
        return tensors
    }
}
