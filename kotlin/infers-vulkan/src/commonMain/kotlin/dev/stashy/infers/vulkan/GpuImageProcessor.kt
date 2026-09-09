package dev.stashy.infers.vulkan

import dev.stashy.infers.*
import dev.stashy.infers.ffi.createGpuImageProcessor
import dev.stashy.infers.internal.CloseGate
import dev.stashy.infers.internal.fromFfi
import dev.stashy.infers.internal.toFfi
import dev.stashy.infers.internal.withFfiErrors
import dev.stashy.infers.ffi.GpuImageProcessor as FfiGpuImageProcessor

/** GPU image preprocessor producing [GpuPending] commits. */
public class GpuImageProcessor private constructor(
    @InfersInternalApi internal val handle: FfiGpuImageProcessor,
) : ImageProcessor<GpuDevice> {
    private val gate = CloseGate("GpuImageProcessor")

    public constructor(device: GpuDevice) : this(
        withFfiErrors { createGpuImageProcessor(device.handle) },
    )

    override val deviceInfo: DeviceInfo
        get() {
            gate.ensureOpen()
            return handle.deviceInfo().fromFfi()
        }

    @InfersInternalApi
    override suspend fun processInternal(
        deferred: Deferred<GpuDevice>,
        options: ProcessingOptions,
    ): Pending<GpuDevice> = withFfiErrors {
        gate.ensureOpen()
        val gpu = deferred as GpuDeferred
        gpu.ensureOpen()
        GpuPending(gpu.handle.process(handle, options.toFfi()))
    }

    override fun close() {
        if (gate.markClosed()) {
            handle.close()
        }
    }
}
