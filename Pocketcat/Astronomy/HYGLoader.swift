//
//  HYGLoader.swift
//  Pocketcat
//
//  Created by Amélie Heinrich on 10/05/2026.
//

import Foundation

struct Star {
    var rightAscension: Float
    var declination: Float
    var magnitude: Float
    var bv: Float
}

class HYGLoader {
    
    
    init(url: URL) {
        
    }
    
    func load() -> Buffer {
        var buffer = Buffer(size: 1)
        return buffer
    }
}
