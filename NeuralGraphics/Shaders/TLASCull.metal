//
//  TLASCull.swift
//  NeuralGraphics
//
//  Created by Amélie Heinrich on 19/03/2026.
//

#include "Common/Bindless.h"

[[kernel]]
void cull_tlas(const device SceneBuffer& scene [[buffer(0)]],
               device MTLIndirectAccelerationStructureInstanceDescriptor* instances [[buffer(1)]],
               device atomic_uint* instanceCount [[buffer(2)]],
               uint instanceID [[thread_position_in_grid]])
{
    if (instanceID >= scene.InstanceCount) return;

    // TODO: cull
    SceneInstance instance = scene.Instances[instanceID];
    SceneEntity entity = scene.Entities[instance.EntityIndex];
    bool visible = true;
    if (visible) {
        uint index = atomic_fetch_add_explicit(instanceCount, 1u, memory_order_relaxed);
        
        instances[index].options = MTLAccelerationStructureInstanceOptionOpaque;
        instances[index].userID = instanceID;
        instances[index].accelerationStructureID = instance.blas;
        instances[index].mask = 0xFF;
        instances[index].intersectionFunctionTableOffset = 0;
        for (int i = 0; i < 3; i++) {
            instances[index].transformationMatrix[0][i] = entity.Transform[0][i];
            instances[index].transformationMatrix[1][i] = entity.Transform[1][i];
            instances[index].transformationMatrix[2][i] = entity.Transform[2][i];
            instances[index].transformationMatrix[3][i] = entity.Transform[3][i];
        }
    }
}
