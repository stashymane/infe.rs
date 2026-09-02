package dev.stashy.infers

import dev.stashy.infers.internal.CloseGate
import dev.stashy.infers.internal.toFfi
import dev.stashy.infers.internal.withFfiErrors
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import dev.stashy.infers.ffi.CpuImageProcessor as FfiCpuImageProcessor

/** CPU image preprocessor producing deferred [CpuPending] commits. */
public class CpuImageProcessor private constructor(
    @InfersInternalApi internal val handle: FfiCpuImageProcessor,
    private val dispatcher: CoroutineDispatcher,
) : ImageProcessor<CpuDevice> {
    private val gate = CloseGate("CpuImageProcessor")

    public constructor() : this(FfiCpuImageProcessor(), Dispatchers.Default.limitedParallelism(1))

    @InfersInternalApi
    public constructor(handle: FfiCpuImageProcessor) : this(handle, Dispatchers.Default.limitedParallelism(1))

    override val deviceInfo: DeviceInfo = CpuDevice.info

    @InfersInternalApi
    internal fun ensureOpen() = gate.ensureOpen()

    @InfersInternalApi
    override suspend fun processInternal(
        image: DeviceImage<CpuDevice>,
        options: ProcessingOptions,
    ): Tensor<CpuDevice> = error("use hardware.onCpu().process(processor, options)")

    override fun close() {
        if (gate.markClosed()) {
            handle.close()
        }
    }
}
