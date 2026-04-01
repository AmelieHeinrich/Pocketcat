//
//  FrameManager.swift
//  Pocketcat
//
//  Created by Amélie Heinrich on 02/03/2026.
//

import Metal
internal import QuartzCore

// FrameManager is the top-level owner of the render graph.
//
//  - It *owns* the passes (strong references, lifetime management).
//  - It holds a RenderTimeline (weak handles to passes, swappable).
//  - It holds a RendererController (swappable strategy for driving a frame).
//  - It owns the command buffer ring and handles CPU/GPU synchronisation.
//
// Render flow:
//   FrameManager.render(drawable:)
//     → controller.render(timeline, &context)
//       → context.camera populated, timeline.execute(context)
//         → pass.render(context) for each pass in order

class FrameManager {
    private var frameIndex: UInt64 = 0
    private let maxFramesInFlight: Int = 3
    private var passes: [Pass]? = nil
    private var controller: RendererController
    private let commandBuffers: [CommandBuffer]
    private let resources: ResourceRegistry = ResourceRegistry()
    private let allocators: [GPULinearAllocator]
    private unowned let registry: SettingsRegistry
    private unowned let lightState: LightState

    private var viewportWidth: Int = 1
    private var viewportHeight: Int = 1
    private var lastRenderScale: Float = -1.0

    private var sceneBuffer: SceneBufferBuilder
    private var mobileTimeline: RenderTimeline? = nil
    private var desktopTimeline: RenderTimeline? = nil
    private var pathtracedTimeline: RenderTimeline? = nil

    var scene: RenderScene?

    // Stats
    private let frameStats: FrameStats
    private var savedCounterEntries: [[(name: String, startSlot: Int, endSlot: Int)]]
    private var lastFrameTime: CFAbsoluteTime = CFAbsoluteTimeGetCurrent()
    private var frameTimeMsAccum: Double = 0
    private var fpsAccum: Double = 0
    private var prevCpuTimestamp: UInt64 = 0
    private var prevGpuTimestamp: UInt64 = 0
    private var gpuToCpuFactor: Double = 1.0

    init(registry: SettingsRegistry, lightState: LightState, frameStats: FrameStats) {
        self.registry = registry
        self.lightState = lightState
        self.controller = EditorRendererController(registry: registry)
        self.frameStats = frameStats
        self.savedCounterEntries = Array(repeating: [], count: 3)

        self.commandBuffers = (0..<3).map { i in
            let cb = CommandBuffer()
            cb.setName(name: "Command Buffer \(i)")
            return cb
        }
        self.allocators = (0..<3).map { i in
            let a = GPULinearAllocator(size: 32 * 1024 * 1024)
            return a
        }
        self.sceneBuffer = SceneBufferBuilder()

        setupTimelines(registry: registry)
    }

    func setScene(_ scene: RenderScene) {
        self.scene = scene
        self.sceneBuffer.build(scene: scene)
        RendererData.residencySet.commit()

        RendererData.waitIdle()
    }

    func resize(width: Int, height: Int) {
        viewportWidth = width
        viewportHeight = height
        let scale = registry.float("Renderer.RenderScale", default: 1.0)
        lastRenderScale = scale
        let scaledW = max(1, Int(Float(width) * scale))
        let scaledH = max(1, Int(Float(height) * scale))
        controller.resize(width: width, height: height)
        for pass in passes! {
            pass.resize(
                renderWidth: scaledW, renderHeight: scaledH, outputWidth: width,
                outputHeight: height)
        }
    }

    func render(drawable: CAMetalDrawable) {
        autoreleasepool {
            // Calibrate GPU clock ticks → CPU nanoseconds via paired timestamp samples
            var cpuTs: MTLTimestamp = 0
            var gpuTs: MTLTimestamp = 0
            (cpuTs, gpuTs) = RendererData.device.sampleTimestamps()
            if prevGpuTimestamp != 0, gpuTs > prevGpuTimestamp {
                let cpuDelta = Double(cpuTs - prevCpuTimestamp)
                let gpuDelta = Double(gpuTs - prevGpuTimestamp)
                if gpuDelta > 0 { gpuToCpuFactor = cpuDelta / gpuDelta }
            }
            prevCpuTimestamp = cpuTs
            prevGpuTimestamp = gpuTs

            // CPU frame timing
            let now = CFAbsoluteTimeGetCurrent()
            let frameTimeMs = (now - lastFrameTime) * 1000.0
            lastFrameTime = now
            frameTimeMsAccum += frameTimeMs
            fpsAccum += frameTimeMs > 0 ? (1000.0 / frameTimeMs) : 0

            let currentScale = registry.float("Renderer.RenderScale", default: 1.0)
            if currentScale != lastRenderScale {
                lastRenderScale = currentScale
                let scaledW = max(1, Int(Float(viewportWidth) * currentScale))
                let scaledH = max(1, Int(Float(viewportHeight) * currentScale))
                for pass in passes! {
                    pass.resize(
                        renderWidth: scaledW, renderHeight: scaledH, outputWidth: viewportWidth,
                        outputHeight: viewportHeight)
                }
            }

            resources.clear()

            let ringIndex = Int(frameIndex) % maxFramesInFlight
            let cmdBuffer = commandBuffers[ringIndex]
            let allocator = allocators[ringIndex]

            // Wait for cmdBuffer to be ready
            RendererData.gpuTimeline.wait(value: frameIndex)

            // Reset per-frame stats accumulation
            FrameAccumulator.current = FrameAccumulator()
            RendererData.counterOffset = ringIndex * RendererData.counterHeapSlotsPerFrame
            RendererData.counterEntries = []

            // Reset, record
            allocator.reset()
            cmdBuffer.begin()

            var context = FrameContext(
                camera: CameraData(),
                cmdBuffer: cmdBuffer,
                drawable: drawable,
                resources: resources,
                scene: scene,
                sceneBuffer: sceneBuffer,
                frameIndex: ringIndex,
                allocator: allocator
            )

            // Update entity transforms before passes run
            if let scene = scene {
                for (i, entity) in scene.entities.enumerated() {
                    sceneBuffer.updateEntityTransform(i, transform: entity.transform)
                }
            }

            switch registry.enum("Renderer.Timeline", as: RendererTimelineType.self, default: .Desktop)
            {
            case .Mobile:
                controller.render(timeline: mobileTimeline!, context: &context)
            case .Desktop:
                controller.render(timeline: desktopTimeline!, context: &context)
            case .Pathtraced:
                controller.render(timeline: pathtracedTimeline!, context: &context)
            }

            // Post-render: let each pass copy current textures to history
            let postEncoder = cmdBuffer.beginComputePass(name: "Post Render")
            postEncoder.consumerBarrier(before: .blit, after: .all)
            for pass in passes! {
                pass.postRender(encoder: postEncoder)
            }
            postEncoder.end()

            // Update lights after updateCamera() has advanced currentFrameIndex
            sceneBuffer.updateLights(sun: lightState.gpuSun(), pointLights: lightState.gpuPointLights())

            // Snapshot accumulator state before commit (render-thread data)
            let frameAccumSnapshot = FrameAccumulator.current
            let counterEntriesSnapshot = RendererData.counterEntries

            // Commit
            cmdBuffer.end()
            cmdBuffer.commit()

            RendererData.cmdQueue.signalEvent(RendererData.gpuTimeline.event, value: frameIndex + 1)
            RendererData.cmdQueue.waitForEvent(
                RendererData.gpuTimeline.event, value: frameIndex + 1)

            // GPU work for this frame is now done — resolve counter heap timestamps
            let gpuTimings = resolveCounterHeap(
                ringIndex: ringIndex,
                accumulator: frameAccumSnapshot,
                counterEntries: counterEntriesSnapshot,
                gpuToCpuFactor: gpuToCpuFactor)

            frameIndex += 1

            RendererData.cmdQueue.waitForDrawable(drawable)
            RendererData.cmdQueue.signalDrawable(drawable)
            drawable.present()

            // Publish stats every 16 frames (~8 Hz at 120 fps)
            if frameIndex % 16 == 0 {
                let avgFrameMs = frameTimeMsAccum / 16.0
                let avgFps     = fpsAccum / 16.0
                frameTimeMsAccum = 0
                fpsAccum = 0

                let scale = Float(currentScale)
                let rw = max(1, Int(Float(viewportWidth)  * scale))
                let rh = max(1, Int(Float(viewportHeight) * scale))
                let allocatedMB = Double(RendererData.device.currentAllocatedSize) / (1024.0 * 1024.0)

                let timelineName: String
                switch registry.enum("Renderer.Timeline", as: RendererTimelineType.self, default: .Desktop) {
                case .Desktop:   timelineName = "Desktop"
                case .Pathtraced: timelineName = "Pathtraced"
                case .Mobile:    timelineName = "Mobile"
                }

                let snapshot = FrameSnapshot(
                    frameTimeMs: avgFrameMs,
                    fps: avgFps,
                    renderWidth: rw,
                    renderHeight: rh,
                    outputWidth: viewportWidth,
                    outputHeight: viewportHeight,
                    renderScale: scale,
                    activeTimeline: timelineName,
                    passTimings: gpuTimings,
                    executeIndirectCount: frameAccumSnapshot.executeIndirectCount,
                    directDrawCount: frameAccumSnapshot.directDrawCount,
                    computeDispatchCount: frameAccumSnapshot.computeDispatchCount,
                    gpuAllocatedMB: allocatedMB
                )

                let stats = self.frameStats
                DispatchQueue.main.async {
                    stats.update(from: snapshot)
                }
            }

            Input.shared.beginFrame()
        }
    }

    // MARK: - Counter Heap Resolution

    private func resolveCounterHeap(
        ringIndex: Int,
        accumulator: FrameAccumulator,
        counterEntries: [(name: String, startSlot: Int, endSlot: Int)],
        gpuToCpuFactor: Double
    ) -> [PassTimingSample] {
        let cpuOnly = accumulator.passRecords.map {
            PassTimingSample(name: $0.name, cpuMs: $0.cpuMs, gpuMs: 0)
        }

        guard let heap = RendererData.counterHeap else { return cpuOnly }

        let baseSlot = ringIndex * RendererData.counterHeapSlotsPerFrame
        let slotRange = baseSlot..<(baseSlot + RendererData.counterHeapSlotsPerFrame)

        guard let data = try? heap.resolveCounterRange(slotRange) else { return cpuOnly }

        // MTL4 timestamp heap entries are GPU clock ticks, not nanoseconds.
        // gpuToCpuFactor (derived from device.sampleTimestamps) converts ticks → CPU ns;
        // dividing by 1_000_000 then gives milliseconds.
        return data.withUnsafeBytes { rawPtr in
            let entries = rawPtr.bindMemory(to: MTL4TimestampHeapEntry.self)
            let invalid: UInt64 = 0xFFFF_FFFF_FFFF_FFFF

            return accumulator.passRecords.map { record in
                var totalGpuMs = 0.0
                for entryIdx in record.gpuStartEntry..<record.gpuEndEntry {
                    guard entryIdx < counterEntries.count else { continue }
                    let entry = counterEntries[entryIdx]
                    guard entry.endSlot >= 0 else { continue }
                    let si = entry.startSlot - baseSlot
                    let ei = entry.endSlot   - baseSlot
                    guard si >= 0, ei < entries.count else { continue }
                    let startNs = entries[si].timestamp
                    let endNs   = entries[ei].timestamp
                    guard startNs != invalid, endNs != invalid, endNs > startNs else { continue }
                    totalGpuMs += Double(endNs - startNs) * gpuToCpuFactor / 1_000_000.0
                }
                return PassTimingSample(name: record.name, cpuMs: record.cpuMs, gpuMs: totalGpuMs)
            }
        }
    }

    // MARK: - Timeline Setup

    func setupTimelines(registry: SettingsRegistry) {
        // Register global settings first so they appear at the top of the UI
        registry.register(
            enum: "Renderer.Timeline", label: "Timeline", default: RendererTimelineType.Desktop)
        registry.register(
            float: "Renderer.RenderScale", label: "Render Scale", default: 0.5, range: 0.25...1.0,
            step: 0.05)

        // Initialize passes
        let cullViewPass = CullViewPass(registry: registry)
        let visibilityPass = VisibilityBufferPass(registry: registry)
        let gbufferPass = GBufferPass()
        let tonemap = TonemapPass(registry: registry)
        let debug = DebugPass.shared
        let upscaler = MetalFXUpscalePass(registry: registry)
        let tlas = TLASBuildPass(settings: registry)
        let pathtracer = Pathtracer(settings: registry)
        let deferred = DeferredPass()
        let accumulationDenoiser = AccumulationDenoiserPass()
        let rtgi = RTGI(settings: registry)
        let rtshadows = RTShadows(settings: registry)
        let rtao = RTAO(settings: registry)
        let rtreflections = RTReflections(settings: registry)
        registry.register(bool: "Debug.DepthTest", label: "Depth Test", default: false)
        debug.registry = registry

        self.passes = [
            tlas, cullViewPass, visibilityPass, pathtracer, tonemap, upscaler, debug, gbufferPass,
            deferred, accumulationDenoiser, rtgi, rtshadows, rtao, rtreflections,
        ]

        // Desktop pipeline
        let desktopTimeline = RenderTimeline()
        desktopTimeline.addPass(tlas)
        desktopTimeline.addPass(cullViewPass)
        desktopTimeline.addPass(visibilityPass)
        desktopTimeline.addPass(gbufferPass)
        desktopTimeline.addPass(rtgi)
        desktopTimeline.addPass(rtshadows)
        desktopTimeline.addPass(rtao)
        desktopTimeline.addPass(rtreflections)
        desktopTimeline.addPass(deferred)
        desktopTimeline.addPass(tonemap)
        desktopTimeline.addPass(upscaler)
        desktopTimeline.addPass(debug)

        // Pathtrace pipeline
        let pathtraceTimeline = RenderTimeline()
        pathtraceTimeline.addPass(tlas)
        pathtraceTimeline.addPass(cullViewPass)
        pathtraceTimeline.addPass(visibilityPass)
        pathtraceTimeline.addPass(gbufferPass)
        pathtraceTimeline.addPass(pathtracer)
        pathtraceTimeline.addPass(accumulationDenoiser)
        pathtraceTimeline.addPass(tonemap)
        pathtraceTimeline.addPass(upscaler)
        pathtraceTimeline.addPass(debug)

        self.desktopTimeline = desktopTimeline
        self.pathtracedTimeline = pathtraceTimeline
    }
}
