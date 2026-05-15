//
//  HYGLoader.swift
//  Pocketcat
//
//  Created by Amélie Heinrich on 10/05/2026.
//

import Foundation

// Must match StarEntry in StarCatalogCompressor.h (16 bytes, GPU-safe layout)
struct Star {
    var rightAscension: Float   // radians
    var declination: Float      // radians
    var magnitude: Float
    var bv: Float               // B-V color index
}

private struct StarCatalogHeader {
    var count: UInt32
}

class HYGLoader {
    private let url: URL

    init(url: URL) {
        self.url = url
    }

    func load() -> (buffer: Buffer, count: Int) {
        guard let data = try? Data(contentsOf: url) else {
            print("[HYGLoader] Failed to read \(url.lastPathComponent)")
            return (Buffer(size: MemoryLayout<Star>.stride), 0)
        }

        let headerSize = MemoryLayout<StarCatalogHeader>.size
        guard data.count >= headerSize else {
            print("[HYGLoader] File too small")
            return (Buffer(size: MemoryLayout<Star>.stride), 0)
        }

        let count = Int(data.withUnsafeBytes { $0.load(as: StarCatalogHeader.self) }.count)
        let payloadSize = count * MemoryLayout<Star>.stride

        guard data.count >= headerSize + payloadSize else {
            print("[HYGLoader] Truncated star catalog")
            return (Buffer(size: MemoryLayout<Star>.stride), 0)
        }

        let buffer = data.withUnsafeBytes { raw -> Buffer in
            let ptr = raw.baseAddress!.advanced(by: headerSize)
            return Buffer(bytes: ptr, size: payloadSize)
        }

        print("[HYGLoader] Loaded \(count) stars")
        return (buffer, count)
    }
}
