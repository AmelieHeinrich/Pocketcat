//
//  TextureLoader.swift
//  Neural Graphics
//
//  Created by Amélie Heinrich on 25/02/2026.
//

import Metal
import MetalKit

class TextureLoader {
    private let loader: MTKTextureLoader

    init(device: MTLDevice) {
        self.loader = MTKTextureLoader(device: device)
    }

    func load(_ name: String, sRGB: Bool = false) throws -> MTLTexture {
        let options: [MTKTextureLoader.Option: Any] = [
            .textureUsage: MTLTextureUsage.shaderRead.rawValue,
            .textureStorageMode: MTLStorageMode.shared.rawValue,
            .SRGB: sRGB,
            .generateMipmaps: true,
            .allocateMipmaps: true,
        ]
        return try loader.newTexture(name: name, scaleFactor: 1.0,
                                     bundle: .main, options: options)
    }

    func load(url: URL, sRGB: Bool = false) throws -> MTLTexture {
        let options: [MTKTextureLoader.Option: Any] = [
            .textureUsage: MTLTextureUsage.shaderRead.rawValue,
            .textureStorageMode: MTLStorageMode.shared.rawValue,
            .SRGB: sRGB,
            .generateMipmaps: true,
            .allocateMipmaps: true
        ]
        return try loader.newTexture(URL: url, options: options)
    }
}
