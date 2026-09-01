package dev.stashy.infers

context(scope: InferenceScope)
public suspend fun <D : Device> ModelSession<D>.infer(input: Tensor<D>): List<CpuTensor> = infer(listOf(input))

@OptIn(InfersInternalApi::class)
context(scope: InferenceScope)
public suspend fun <D : Device> ModelSession<D>.infer(inputs: List<Tensor<D>>): List<CpuTensor> =
    runInternal(inputs).map { scope.register(it) }

@OptIn(InfersInternalApi::class)
context(scope: InferenceScope)
public suspend fun <D : Device> ImageProcessor<D>.process(
    image: DeviceImage<D>,
    options: ProcessingOptions,
): Tensor<D> = scope.register(processInternal(image, options))

@OptIn(InfersInternalApi::class)
context(scope: InferenceScope)
public suspend fun <D : Device> ImageProcessor<D>.process(
    image: DeviceImage<D>,
    builder: ProcessingOptionsBuilder.() -> Unit,
): Tensor<D> = scope.register(processInternal(image, ProcessingOptions.invoke(builder)))
