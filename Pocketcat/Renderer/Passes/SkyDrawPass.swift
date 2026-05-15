//
//  SkyDrawPass.swift
//  Pocketcat
//
//  Created by Amélie Heinrich on 03/04/2026.
//

import Metal
import simd

class SkyDrawPass: Pass {
    private let pipeline: ComputePipeline
    private unowned var settings: SettingsRegistry

    init(settings: SettingsRegistry) {
        self.settings = settings
        pipeline = ComputePipeline(function: "sky_draw", name: "Sky Draw")
        super.init()
    }

    override func render(context: FrameContext) {
        let skyCubemap = context.resources.get("Sky.Cubemap") as Texture?
        let depth = context.resources.get("GBuffer.Depth") as Texture?
        let hdr = context.resources.get("HDR") as Texture?

        guard let skyCubemap, let depth, let hdr else { return }

        let w = hdr.texture.width
        let h = hdr.texture.height

        let cp = context.cmdBuffer.beginComputePass(name: "Sky : Draw")
        cp.consumerBarrier(before: .dispatch, after: .dispatch)
        cp.setPipeline(pipeline: pipeline)
        cp.setBuffer(buf: context.sceneBuffer.buffer, index: 0)
        cp.setTexture(texture: depth, index: 0)
        cp.setTexture(texture: skyCubemap, index: 1)
        cp.setTexture(texture: hdr, index: 2)
        cp.dispatch(threads: MTLSizeMake((w + 7) / 8, (h + 7) / 8, 1), threadsPerGroup: MTLSizeMake(8, 8, 1))
        cp.end()
    }
}
