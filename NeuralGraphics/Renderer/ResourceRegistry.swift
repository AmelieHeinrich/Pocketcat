//
//  ResourceRegistry.swift
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 02/03/2026.
//

// ResourceRegistry is the shared blackboard that passes use to publish and
// consume named resources within a single frame.
//
// Usage pattern:
//   GBufferPass  → context.resources.register(normalsTexture, for: "GBuffer.Normals")
//   SSAOPass     → let normals: Texture? = context.resources.get("GBuffer.Normals")
//   NNAOController → fetch GBuffer attachments + RTAO output after timeline.execute()
//
// The registry is a class so reference semantics apply: passes can register
// resources even when FrameContext is passed by value, and the mutations are
// immediately visible to all other holders.
//
// FrameManager clears the registry at the top of each frame so there is never
// any bleed-through of stale resources from a prior frame.

class ResourceRegistry {
    private var resources: [String: Any] = [:]

    func register<T>(_ resource: T, for key: String) {
        resources[key] = resource
    }

    func get<T>(_ key: String) -> T? {
        resources[key] as? T
    }

    func clear() {
        resources.removeAll(keepingCapacity: true)
    }
}
