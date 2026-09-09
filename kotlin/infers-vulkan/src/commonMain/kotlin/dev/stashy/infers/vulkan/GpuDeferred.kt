package dev.stashy.infers.vulkan

import dev.stashy.infers.DataType
import dev.stashy.infers.Deferred
import dev.stashy.infers.DeviceImage
import dev.stashy.infers.DeviceInfo
import dev.stashy.infers.InfersInternalApi
import dev.stashy.infers.Pending
import dev.stashy.infers.Tensor
import dev.stashy.infers.TensorShape
import dev.stashy.infers.internal.CloseGate
import dev.stashy.infers.internal.fromFfi
import dev.stashy.infers.internal.withFfiErrors
import dev.stashy.infers.ffi.GpuDeferred as FfiGpuDeferred
import dev.stashy.infers.ffi.GpuPending as FfiGpuPending

/** GPU-deferred hardware image placement. */
public class GpuDeferred @InfersInternalApi constructor(
    internal val handle: FfiGpuDeferred,
) : Deferred<GpuDevice> {
    private val gate = CloseGate("GpuDeferred")

    public val deviceInfo: DeviceInfo
        get() = withFfiErrors {
            gate.ensureOpen()
            handle.deviceInfo().fromFfi()
        }

    @InfersInternalApi
    internal fun ensureOpen() = gate.ensureOpen()

    @InfersInternalApi
    override suspend fun materializeInternal(): DeviceImage<GpuDevice> = withFfiErrors {
        gate.ensureOpen()
        GpuImage(handle.materialize())
    }

    override fun close() {
        if (gate.markClosed()) {
            handle.close()
        }
    }
}

/** GPU pending preprocess commit. */
public class GpuPending @InfersInternalApi constructor(
    internal val handle: FfiGpuPending,
) : Pending<GpuDevice> {
    private val gate = CloseGate("GpuPending")

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
    override suspend fun materializeInternal(): Tensor<GpuDevice> = withFfiErrors {
        gate.ensureOpen()
        GpuTensor.fromFfi(handle.materialize())
    }

    override fun close() {
        if (gate.markClosed()) {
            handle.close()
        }
    }
}
