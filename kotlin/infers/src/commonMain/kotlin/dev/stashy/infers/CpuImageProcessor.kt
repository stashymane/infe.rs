package dev.stashy.infers

import dev.stashy.infers.internal.CloseGate
import dev.stashy.infers.internal.toFfi
import dev.stashy.infers.internal.withFfiErrors
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import dev.stashy.infers.ffi.CpuImageProcessor as FfiCpuImageProcessor

/** CPU image preprocessor producing [CpuTensor]. */
public class CpuImageProcessor private constructor(
    internal val handle: FfiCpuImageProcessor,
    private val dispatcher: CoroutineDispatcher,
) : ImageProcessor<CpuDevice> {
    private val gate = CloseGate("CpuImageProcessor")

    public constructor() : this(FfiCpuImageProcessor(), Dispatchers.Default.limitedParallelism(1))

    @InfersInternalApi
    public constructor(handle: FfiCpuImageProcessor) : this(handle, Dispatchers.Default.limitedParallelism(1))

    override val deviceInfo: DeviceInfo = CpuDevice.info

    @InfersInternalApi
    override suspend fun processInternal(
        image: DeviceImage<CpuDevice>,
        options: ProcessingOptions,
    ): Tensor<CpuDevice> = withContext(dispatcher) {
        withFfiErrors {
            gate.ensureOpen()
            val host = image as HostImage
            CpuTensor.fromFfi(handle.process(host.handle, options.toFfi()))
        }
    }

    override fun close() {
        if (gate.markClosed()) {
            handle.close()
        }
    }
}
