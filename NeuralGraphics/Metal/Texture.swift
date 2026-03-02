//
//  Texture.swift
//  Neural Graphics
//
//  Created by Amélie Heinrich on 25/02/2026.
//

import Metal

class Texture {
    var texture: MTLTexture
    var descriptor: MTLTextureDescriptor
    var allocated: Bool = false
    var label: String = ""
    
    init(descriptor: MTLTextureDescriptor) {
        self.descriptor = descriptor
        self.texture = RendererData.device.makeTexture(descriptor: descriptor)!
        self.allocated = true
        
        RendererData.residencySet.addAllocation(self.texture)
    }
    
    init(texture: MTLTexture, descriptor: MTLTextureDescriptor) {
        self.descriptor = descriptor
        self.texture = texture
    }
    
    deinit {
        if self.allocated {
            RendererData.residencySet.removeAllocation(self.texture)
        }
    }
    
    func setLabel(name: String) {
        self.texture.label = name
        self.label = name
    }
    
    func resize(width: Int, height: Int, computeMipLevels: Bool = false) {
        RendererData.residencySet.removeAllocation(self.texture)
        
        self.descriptor.width = width
        self.descriptor.height = height
        if computeMipLevels {
            let maxDim = width > height ? width : height
            let mipLevels = Int(log2(Double(maxDim))) + 1
            self.descriptor.mipmapLevelCount = mipLevels
        }
        
        self.texture = RendererData.device.makeTexture(descriptor: self.descriptor)!
        self.texture.label = self.label
        
        RendererData.residencySet.addAllocation(self.texture)
    }
    
    func uploadData(region: MTLRegion, mip: Int, data: UnsafeRawPointer, bpp: Int) {
        self.texture.replace(region: region, mipmapLevel: mip, withBytes: data, bytesPerRow: self.texture.width * bpp)
    }
}
