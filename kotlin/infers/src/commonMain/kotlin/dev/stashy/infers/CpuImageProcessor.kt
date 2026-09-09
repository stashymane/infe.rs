package dev.stashy.infers

import dev.stashy.infers.internal.CloseGate
import dev.stashy.infers.internal.toFfi
import dev.stashy.infers.internal.withFfiErrors
import dev.stashy.infers.ffi.CpuImageProcessor as FfiCpuImageProcessor

/** CPU image preprocessor producing deferred [CpuPending] commits. */
public class CpuImageProcessor @InfersInternalApi constructor(
    @InfersInternalApi internal val handle: FfiCpuImageProcessor,
) : ImageProcessor<CpuDevice> {
    private val gate = CloseGate("CpuImageProcessor")

    public constructor() : this(FfiCpuImageProcessor())

    override val deviceInfo: DeviceInfo = CpuDevice.info

    @InfersInternalApi
    override suspend fun processInternal(
        deferred: Deferred<CpuDevice>,
        options: ProcessingOptions,
    ): Pending<CpuDevice> = withFfiErrors {
        gate.ensureOpen()
        val cpu = deferred as CpuDeferred
        cpu.ensureOpen()
        CpuPending(cpu.handle.process(handle, options.toFfi()))
    }

    override fun close() {
        if (gate.markClosed()) {
            handle.close()
        }
    }
}
