//
//  TextureLoader.swift
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 02/03/2026.
//

import Metal
import Foundation

class TextureLoader {
    // Mirrors the C TextureHeader from TextureCompressor.h.
    // All fields are uint32_t — no padding, no platform-width surprises.
    // Binary layout:
    //   offset  0: format    (uint32_t → 4 bytes)
    //   offset  4: width     (uint32_t → 4 bytes)
    //   offset  8: height    (uint32_t → 4 bytes)
    //   offset 12: mipLevels (uint32_t → 4 bytes)
    //   total: 16 bytes
    private static let headerSize = 16

    private static func astcMipSize(width: Int, height: Int) -> Int {
        let blockW = (width  + 3) / 4
        let blockH = (height + 3) / 4
        return blockW * blockH * 16
    }

    static func load(resource: String, withExtension ext: String = "astc", label: String = "") -> Texture? {
        guard let url = Bundle.main.url(forResource: resource, withExtension: ext) else {
            print("[TextureLoader] Not found in bundle: \(resource).\(ext)")
            return nil
        }
        return load(url: url, label: label)
    }

    static func load(url: URL, label: String = "") -> Texture? {
        guard let data = try? Data(contentsOf: url, options: .mappedIfSafe) else {
            print("[TextureLoader] Failed to read: \(url.path)")
            return nil
        }
        guard data.count >= headerSize else {
            print("[TextureLoader] File too small: \(url.path)")
            return nil
        }

        // Parse header fields individually to avoid Swift struct layout assumptions.
        let formatRaw = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 0,  as: UInt32.self) }
        let width     = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 4,  as: UInt32.self) }
        let height    = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 8,  as: UInt32.self) }
        let mipLevels = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 12, as: UInt32.self) }

        guard let pixelFormat = MTLPixelFormat(rawValue: UInt(formatRaw)) else {
            print("[TextureLoader] Unknown pixel format \(formatRaw): \(url.path)")
            return nil
        }

        let w    = Int(width)
        let h    = Int(height)
        let mips = Int(mipLevels)

        // .shared storage allows CPU uploads via replace(region:…) without a blit pass.
        // On Apple Silicon (unified memory) this has no performance cost vs .private.
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: pixelFormat,
            width: w,
            height: h,
            mipmapped: mips > 1
        )
        desc.mipmapLevelCount = mips
        desc.storageMode = .shared
        desc.usage = .shaderRead

        // Don't add to the residency set here — TextureLoader may be called
        // from a background thread. The caller is responsible for calling
        // makeResident() once back on the main thread.
        let texture = Texture(descriptor: desc, makeResident: false)
        if !label.isEmpty { texture.setLabel(name: label) }

        var offset = headerSize
        var mipW = w
        var mipH = h

        for mip in 0..<mips {
            let blockW   = (mipW + 3) / 4
            let mipBytes = astcMipSize(width: mipW, height: mipH)
            let bpr      = blockW * 16  // bytes per block-row

            guard offset + mipBytes <= data.count else {
                print("[TextureLoader] Truncated at mip \(mip): \(url.path)")
                break
            }

            data.withUnsafeBytes { raw in
                let ptr    = raw.baseAddress!.advanced(by: offset)
                let region = MTLRegionMake2D(0, 0, mipW, mipH)
                texture.uploadCompressedData(region: region, mip: mip, data: ptr, bytesPerRow: bpr)
            }

            offset += mipBytes
            mipW = max(1, mipW / 2)
            mipH = max(1, mipH / 2)
        }

        return texture
    }
}
