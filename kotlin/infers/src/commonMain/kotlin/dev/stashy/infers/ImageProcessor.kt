package dev.stashy.infers

/** Image preprocessor producing [Tensor] on device [D]. */
public interface ImageProcessor<D : Device> : AutoCloseable {
    public val deviceInfo: DeviceInfo

    @InfersInternalApi
    public suspend fun processInternal(image: DeviceImage<D>, options: ProcessingOptions): Tensor<D>
}
