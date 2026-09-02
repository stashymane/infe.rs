package dev.stashy.infers.vulkan

import dev.stashy.infers.CpuTensor
import dev.stashy.infers.DeviceInfo
import dev.stashy.infers.InfersInternalApi
import dev.stashy.infers.ModelSession
import dev.stashy.infers.Pending
import dev.stashy.infers.Tensor
import dev.stashy.infers.TensorShape
import dev.stashy.infers.internal.CloseGate
import dev.stashy.infers.internal.fromFfi
import dev.stashy.infers.internal.withFfiErrors
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.withContext
import dev.stashy.infers.ffi.GpuSession as FfiGpuSession

/** ExecuTorch session running on GPU (Vulkan delegate). */
@OptIn(ExperimentalCoroutinesApi::class)
public class GpuSession @InfersInternalApi constructor(
    private val handle: FfiGpuSession,
    private val dispatcher: CoroutineDispatcher = Dispatchers.Default.limitedParallelism(1),
) : ModelSession<GpuDevice> {
    private val gate = CloseGate("GpuSession")

    override val deviceInfo: DeviceInfo
        get() {
            gate.ensureOpen()
            return handle.deviceInfo().fromFfi()
        }

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
    override suspend fun runInternal(inputs: List<Tensor<GpuDevice>>): List<CpuTensor> = withContext(dispatcher) {
        withFfiErrors {
            gate.ensureOpen()
            val primary = inputs.singleOrNull() as? GpuTensor ?: error("expected one GPU tensor input")
            primary.ensureOpen()
            handle
                .inferTensor(primary.handle)
                .map { CpuTensor.fromFfi(it) }
        }
    }

    @InfersInternalApi
    override suspend fun inferPendingInternal(pending: Pending<GpuDevice>): List<CpuTensor> = withContext(dispatcher) {
        withFfiErrors {
            gate.ensureOpen()
            val gpuPending = pending as GpuPending
            gpuPending.ensureOpen()
            handle.infer(gpuPending.handle).map { CpuTensor.fromFfi(it) }
        }
    }

    override fun close() {
        if (gate.markClosed()) {
            handle.close()
        }
    }
}
