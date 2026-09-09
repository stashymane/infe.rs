package dev.stashy.infers.vulkan

import dev.stashy.infers.*

@OptIn(InfersInternalApi::class)
context(scope: InferenceScope)
public suspend fun GpuDevice.uploadTensor(tensor: CpuTensor): GpuTensor = scope.register(uploadTensorInternal(tensor))
