package dev.stashy.infers.vulkan

import dev.stashy.infers.*

@OptIn(InfersInternalApi::class)
context(scope: InferenceScope)
public suspend fun GpuDevice.uploadTensor(tensor: CpuTensor): GpuTensor = scope.register(uploadTensorInternal(tensor))

@OptIn(InfersInternalApi::class)
context(scope: InferenceScope)
public suspend fun ModelSession<GpuDevice>.infer(pending: GpuPending): List<CpuTensor> =
    inferPendingInternal(pending).map { scope.register(it) }

@OptIn(InfersInternalApi::class)
context(scope: InferenceScope)
public suspend fun ModelSession<GpuDevice>.infer(input: Tensor<GpuDevice>): List<CpuTensor> =
    infer(listOf(input))

@OptIn(InfersInternalApi::class)
context(scope: InferenceScope)
public suspend fun ModelSession<GpuDevice>.infer(inputs: List<Tensor<GpuDevice>>): List<CpuTensor> =
    runInternal(inputs).map { scope.register(it) }
