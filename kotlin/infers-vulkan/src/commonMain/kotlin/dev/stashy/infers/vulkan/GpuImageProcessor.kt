package dev.stashy.infers.vulkan

import dev.stashy.infers.DeviceInfo
import dev.stashy.infers.InfersInternalApi
import dev.stashy.infers.ProcessingOptions
import dev.stashy.infers.ffi.createGpuImageProcessor
import dev.stashy.infers.internal.CloseGate
import dev.stashy.infers.internal.fromFfi
import dev.stashy.infers.internal.toFfi
import dev.stashy.infers.internal.withFfiErrors
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import dev.stashy.infers.ffi.GpuImageProcessor as FfiGpuImageProcessor

/** GPU image preprocessor producing [GpuPending] commits. */
public class GpuImageProcessor private constructor(
    @InfersInternalApi internal val handle: FfiGpuImageProcessor,
    private val dispatcher: CoroutineDispatcher,
) : dev.stashy.infers.ImageProcessor<GpuDevice> {
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
        image: dev.stashy.infers.DeviceImage<GpuDevice>,
        options: ProcessingOptions,
    ): dev.stashy.infers.Tensor<GpuDevice> = error("use hardware.on(device).process(processor, options)")

    override fun close() {
        if (gate.markClosed()) {
            handle.close()
        }
    }
}
