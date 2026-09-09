package dev.stashy.infers

@OptIn(InfersInternalApi::class)
context(scope: InferenceScope)
public suspend fun <D : Device<D>> ModelSession<D>.infer(pending: Pending<D>): List<CpuTensor> =
    inferPendingInternal(pending).map { scope.register(it) }

context(scope: InferenceScope)
public suspend fun <D : Device<D>> ModelSession<D>.infer(input: Tensor<D>): List<CpuTensor> = infer(listOf(input))

@OptIn(InfersInternalApi::class)
context(scope: InferenceScope)
public suspend fun <D : Device<D>> ModelSession<D>.infer(inputs: List<Tensor<D>>): List<CpuTensor> =
    runInternal(inputs).map { scope.register(it) }
