package dev.stashy.infers

import dev.stashy.infers.internal.toFfi
import dev.stashy.infers.internal.withFfiErrors

@OptIn(InfersInternalApi::class)
context(scope: InferenceScope)
public suspend fun CpuPending.materialize(): CpuTensor = scope.register(
    withFfiErrors {
        ensureOpen()
        CpuTensor.fromFfi(handle.materialize())
    },
)

@OptIn(InfersInternalApi::class)
context(scope: InferenceScope)
public suspend fun CpuDeferred.process(processor: CpuImageProcessor, options: ProcessingOptions): CpuPending =
    scope.register(
        withFfiErrors {
            ensureOpen()
            processor.ensureOpen()
            CpuPending(handle.process(processor.handle, options.toFfi()))
        },
    )

@OptIn(InfersInternalApi::class)
context(scope: InferenceScope)
public suspend fun ModelSession<CpuDevice>.infer(pending: CpuPending): List<CpuTensor> =
    inferPendingInternal(pending).map { scope.register(it) }

@OptIn(InfersInternalApi::class)
context(scope: InferenceScope)
public suspend fun ModelSession<CpuDevice>.infer(input: Tensor<CpuDevice>): List<CpuTensor> = infer(listOf(input))

@OptIn(InfersInternalApi::class)
context(scope: InferenceScope)
public suspend fun ModelSession<CpuDevice>.infer(inputs: List<Tensor<CpuDevice>>): List<CpuTensor> =
    runInternal(inputs).map { scope.register(it) }
