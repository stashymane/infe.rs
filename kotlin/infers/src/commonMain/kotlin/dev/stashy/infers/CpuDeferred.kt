package dev.stashy.infers

import dev.stashy.infers.internal.CloseGate
import dev.stashy.infers.internal.fromFfi
import dev.stashy.infers.internal.withFfiErrors
import dev.stashy.infers.ffi.CpuDeferred as FfiCpuDeferred

/** CPU-deferred hardware image placement. */
public class CpuDeferred @InfersInternalApi constructor(
    internal val handle: FfiCpuDeferred,
) : Deferred<CpuDevice> {
    private val gate = CloseGate("CpuDeferred")

    @InfersInternalApi
    internal fun ensureOpen() = gate.ensureOpen()

    @InfersInternalApi
    override suspend fun materializeInternal(): DeviceImage<CpuDevice> = withFfiErrors {
        gate.ensureOpen()
        CpuImage(handle.materialize())
    }

    override fun close() {
        if (gate.markClosed()) {
            handle.close()
        }
    }
}

/** CPU pending preprocess commit. */
public class CpuPending @InfersInternalApi constructor(
    internal val handle: dev.stashy.infers.ffi.CpuPending,
) : Pending<CpuDevice> {
    private val gate = CloseGate("CpuPending")

    override val shape: TensorShape
        get() = withFfiErrors {
            gate.ensureOpen()
            handle.shape().fromFfi()
        }

    override val dtype: DataType
        get() = withFfiErrors {
            gate.ensureOpen()
            handle.dtype().fromFfi()
        }

    @InfersInternalApi
    internal fun ensureOpen() = gate.ensureOpen()

    @InfersInternalApi
    override suspend fun materializeInternal(): Tensor<CpuDevice> = withFfiErrors {
        gate.ensureOpen()
        CpuTensor.fromFfi(handle.materialize())
    }

    override fun close() {
        if (gate.markClosed()) {
            handle.close()
        }
    }
}
