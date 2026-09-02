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

    override fun close() {
        if (gate.markClosed()) {
            handle.close()
        }
    }
}
