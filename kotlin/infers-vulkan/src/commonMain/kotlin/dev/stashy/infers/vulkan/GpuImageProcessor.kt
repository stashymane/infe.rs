package dev.stashy.infers.vulkan

import dev.stashy.infers.DeviceImage
import dev.stashy.infers.DeviceInfo
import dev.stashy.infers.ImageProcessor
import dev.stashy.infers.InfersInternalApi
import dev.stashy.infers.ProcessingOptions
import dev.stashy.infers.Tensor
import dev.stashy.infers.ffi.createGpuImageProcessor
import dev.stashy.infers.internal.CloseGate
import dev.stashy.infers.internal.fromFfi
import dev.stashy.infers.internal.toFfi
import dev.stashy.infers.internal.withFfiErrors
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import dev.stashy.infers.ffi.GpuImageProcessor as FfiGpuImageProcessor

/** GPU image preprocessor producing [GpuTensor]. */
public class GpuImageProcessor private constructor(
    internal val handle: FfiGpuImageProcessor,
    private val dispatcher: CoroutineDispatcher,
) : ImageProcessor<GpuDevice> {
    private val gate = CloseGate("GpuImageProcessor")

    public constructor(device: GpuDevice) : this(
        withFfiErrors { createGpuImageProcessor(device.handle) },
        Dispatchers.Default.limitedParallelism(1),
    )

    override val deviceInfo: DeviceInfo
        get() {
            gate.ensureOpen()
            return handle.deviceInfo().fromFfi()
        }

    @InfersInternalApi
    internal fun ensureOpen() = gate.ensureOpen()

    @InfersInternalApi
    override suspend fun processInternal(
        image: DeviceImage<GpuDevice>,
        options: ProcessingOptions,
    ): Tensor<GpuDevice> = withContext(dispatcher) {
        withFfiErrors {
            gate.ensureOpen()
            val gpuImage = image as GpuImage
            GpuTensor.fromFfi(handle.process(gpuImage.handle, options.toFfi()))
        }
    }

    override fun close() {
        if (gate.markClosed()) {
            handle.close()
        }
    }
}
