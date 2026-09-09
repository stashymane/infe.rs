package dev.stashy.infers

/** Unmaterialized image placement on device [D]. */
public interface Deferred<D : Device<D>> : AutoCloseable {
    @InfersInternalApi
    public suspend fun materializeInternal(): DeviceImage<D>
}

/** Unmaterialized preprocess/tensor work on device [D]. */
public interface Pending<D : Device<D>> : AutoCloseable {
    public val shape: TensorShape
    public val dtype: DataType

    @InfersInternalApi
    public suspend fun materializeInternal(): Tensor<D>
}

@OptIn(InfersInternalApi::class)
context(scope: InferenceScope)
public suspend fun <D : Device<D>> Deferred<D>.materialize(): DeviceImage<D> = scope.register(materializeInternal())

@OptIn(InfersInternalApi::class)
context(scope: InferenceScope)
public suspend fun <D : Device<D>> Pending<D>.materialize(): Tensor<D> = scope.register(materializeInternal())
