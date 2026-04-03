# Pocketcat

![](.github/reg.png)
![](.github/ptbistro.png)
![](.github/pt.png)

This repository showcases various modern rendering techniques implemented using Metal 4 API, focused on GPU driven rendering, raytracing and neural graphics.

## Building and running

This project works on any M3+ GPU. Anything below is not supported due to lack of support for indirect mesh ICB and Metal 4 raytracing.\
To build, just open the project in Xcode and run the Pocketcat scheme.\
To add more meshes, you can add any GLTF model in SourceAssets and run the AssetBaker program to bake everything. The engine does not support runtime creation of scenes yet, you have to declare them programmatically in Pocketcat/Core/Scene.swift.

## Current features

- GPU driven debug renderer
- GPU driven TLAS build
- GPU driven visibility buffer with mesh shaders
- MetalFX spatial/temporal upscaling
- Stochastic reference pathtracer
- Hillaire sky model

## WIP

All of these features are complete on the raytracing part but are missing the denoiser I am currently working on:
- Raytraced sun shadows
- Raytraced ambient occlusion
- Raytraced global illumination
- Raytraced reflections

## TODO

- Proper meshlet culling and LOD selection
- Inference engine
- SSAO + NNAO
- Point lights in raster and PT
- Clustered light culling
- Volumetric clouds
- Neural sky model?
- Auto-exposure
- Wavefront pathtracer with custom materials
