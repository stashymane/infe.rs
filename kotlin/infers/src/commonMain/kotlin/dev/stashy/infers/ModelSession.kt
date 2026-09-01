package dev.stashy.infers

import dev.stashy.infers.internal.CloseGate
import dev.stashy.infers.internal.fromFfi
import dev.stashy.infers.internal.withFfiErrors
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.withContext
import dev.stashy.infers.ffi.CpuSession as FfiCpuSession

/** Loaded model ready for inference on device [D]. */
public interface ModelSession<D : Device> : AutoCloseable {
    public val deviceInfo: DeviceInfo
    public val inputShapes: List<TensorShape>
    public val outputShapes: List<TensorShape>

    @InfersInternalApi
    public suspend fun runInternal(inputs: List<Tensor<D>>): List<CpuTensor>
}

/**
 * ExecuTorch session running on CPU (XNNPACK).
 *
 * Each session owns a single-threaded dispatcher matching the native mutex.
 */
@OptIn(ExperimentalCoroutinesApi::class)
public class CpuSession @InfersInternalApi constructor(
    private val handle: FfiCpuSession,
    private val dispatcher: CoroutineDispatcher = Dispatchers.Default.limitedParallelism(1),
) : ModelSession<CpuDevice> {
    private val gate = CloseGate("CpuSession")

    override val deviceInfo: DeviceInfo = CpuDevice.info

    override val inputShapes: List<TensorShape>
        get() {
            gate.ensureOpen()
            return handle.inputShapes().map { it.fromFfi() }
        }

    override val outputShapes: List<TensorShape>
        get() {
            gate.ensureOpen()
            return handle.outputShapes().map { it.fromFfi() }
        }

    @InfersInternalApi
    override suspend fun runInternal(inputs: List<Tensor<CpuDevice>>): List<CpuTensor> = withContext(dispatcher) {
        withFfiErrors {
            gate.ensureOpen()
            handle
                .run(inputs.map { (it as CpuTensor).handle })
                .map { CpuTensor.fromFfi(it) }
        }
    }

    override fun close() {
        if (gate.markClosed()) {
            handle.close()
        }
    }
}
