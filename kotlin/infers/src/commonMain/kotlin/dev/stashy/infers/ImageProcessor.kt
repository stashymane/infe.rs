package dev.stashy.infers

/** Image preprocessor producing [Pending] commits on device [D]. */
public interface ImageProcessor<D : Device<D>> : AutoCloseable {
    public val deviceInfo: DeviceInfo

    @InfersInternalApi
    public suspend fun processInternal(deferred: Deferred<D>, options: ProcessingOptions): Pending<D>
}

@OptIn(InfersInternalApi::class)
context(scope: InferenceScope)
public suspend fun <D : Device<D>> Deferred<D>.process(
    processor: ImageProcessor<D>,
    options: ProcessingOptions,
): Pending<D> = scope.register(processor.processInternal(this, options))
