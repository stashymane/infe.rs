package dev.stashy.infers.vulkan

import dev.stashy.infers.Device
import dev.stashy.infers.FfiBackedImageProcessor
import dev.stashy.infers.InfersInternalApi
import dev.stashy.infers.ffi.createGpuContext
import dev.stashy.infers.ffi.createGpuImageProcessor
import dev.stashy.infers.internal.CloseGate
import dev.stashy.infers.internal.fromFfi
import dev.stashy.infers.internal.toFfi
import dev.stashy.infers.internal.withFfiErrors
import dev.stashy.infers.ffi.GpuContext as FfiGpuContext

/**
 * Shared Vulkan context. Create before GPU processors or Vulkan inference, and
 * drop sessions/processors before the last context handle.
 */
public class GpuContext private constructor(
    internal val handle: FfiGpuContext,
) : AutoCloseable {
    private val gate = CloseGate("GpuContext")

    public constructor(device: Device) : this(
        withFfiErrors { createGpuContext(device.toFfi()) },
    )

    public val device: Device
        get() {
            gate.ensureOpen()
            return handle.device().fromFfi()
        }

    override fun close() {
        if (gate.markClosed()) {
            handle.close()
        }
    }
}

public class GpuImageProcessor : FfiBackedImageProcessor {
    @OptIn(InfersInternalApi::class)
    public constructor(context: GpuContext) : super(
        withFfiErrors { createGpuImageProcessor(context.handle) },
        "GpuImageProcessor",
    )
}
