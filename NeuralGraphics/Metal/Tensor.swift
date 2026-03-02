//
//  Tensor.swift
//  Neural Graphics
//
//  Created by Amélie Heinrich on 27/02/2026.
//

import Metal

class Tensor {
    var tensor: MTLTensor
    
    init(descriptor: MTLTensorDescriptor) {
        self.tensor = try! RendererData.device.makeTensor(descriptor: descriptor)
        RendererData.residencySet.addAllocation(self.tensor)
    }
    
    deinit {
        RendererData.residencySet.removeAllocation(self.tensor)
    }
    
    func setName(name: String) {
        tensor.label = name
    }
    
    func getAddress() -> MTLResourceID {
        return tensor.gpuResourceID
    }
    
    // TODO(amelie): Copy operations with buffers, textures, etc?
}
