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
    
    init(descriptor: MTLTextureDescriptor, makeResident: Bool = true) {
        self.descriptor = descriptor
        self.texture = RendererData.device.makeTexture(descriptor: descriptor)!

        if makeResident {
            RendererData.residencySet.addAllocation(self.texture)
            self.allocated = true
        }
    }

    /// Adds this texture to the residency set. Must be called on the main thread
    /// (or whichever thread owns the residency set) after background loading finishes.
    func makeResident() {
        guard !allocated else { return }
        RendererData.residencySet.addAllocation(self.texture)
        allocated = true
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

    // For block-compressed formats (ASTC 4x4, BC7, etc.): bytesPerRow is the stride
    // of one row of compressed blocks, not one row of pixels.
    // e.g. ASTC 4x4: bytesPerRow = ((width + 3) / 4) * 16
    func uploadCompressedData(region: MTLRegion, mip: Int, data: UnsafeRawPointer, bytesPerRow: Int) {
        self.texture.replace(region: region, mipmapLevel: mip, withBytes: data, bytesPerRow: bytesPerRow)
    }
}
