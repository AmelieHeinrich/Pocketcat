//
//  ResourceManager.swift
//  Neural Graphics
//
//  Created by Amélie Heinrich on 25/02/2026.
//

import Metal
import MetalKit

class ResourceManager {
    private struct CachedTexture {
        let texture: MTLTexture
        var refCount: Int
    }

    private let textureLoader: ImageLoader
    private var textures: [String: CachedTexture] = [:]

    init(device: MTLDevice) {
        self.textureLoader = ImageLoader(device: device)
        listAvailableAssets()
    }

    // MARK: - Textures

    func texture(named name: String, sRGB: Bool = false) throws -> MTLTexture {
        if let cached = textures[name] {
            textures[name]!.refCount += 1
            return cached.texture
        }
        let tex = try textureLoader.load(name, sRGB: sRGB)
        RendererData.residencySet.addAllocation(tex)
        textures[name] = CachedTexture(texture: tex, refCount: 1)
        return tex
    }

    func texture(url: URL, sRGB: Bool = false) throws -> MTLTexture {
        let key = url.absoluteString
        if let cached = textures[key] {
            textures[key]!.refCount += 1
            return cached.texture
        }
        let tex = try textureLoader.load(url: url, sRGB: sRGB)
        RendererData.residencySet.addAllocation(tex)
        textures[key] = CachedTexture(texture: tex, refCount: 1)
        return tex
    }

    func releaseTexture(named name: String) {
        guard var cached = textures[name] else { return }
        cached.refCount -= 1
        if cached.refCount <= 0 {
            RendererData.residencySet.removeAllocation(cached.texture)
            textures.removeValue(forKey: name)
        } else {
            textures[name] = cached
        }
    }

    // MARK: - Asset discovery

    private func listAvailableAssets() {
        print("[ResourceManager] Available assets:")

        let extensions = ["png", "jpg", "jpeg", "hdr", "astc", "gltf", "mtlpackage"]
        var found = false

        if let resourcePath = Bundle.main.resourcePath {
            let fm = FileManager.default
            if let contents = try? fm.contentsOfDirectory(atPath: resourcePath) {
                for file in contents.sorted() {
                    let ext = (file as NSString).pathExtension.lowercased()
                    if extensions.contains(ext) {
                        print("  [asset] \(file)")
                        found = true
                    }
                }
            }
        }

        if !found {
            print("  (none found)")
        }
    }
}
