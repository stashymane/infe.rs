package dev.stashy.infers.vulkan

import dev.stashy.infers.DataType
import dev.stashy.infers.Deferred
import dev.stashy.infers.DeviceInfo
import dev.stashy.infers.HardwareImage
import dev.stashy.infers.InferenceScope
import dev.stashy.infers.InfersInternalApi
import dev.stashy.infers.Pending
import dev.stashy.infers.ProcessingOptions
import dev.stashy.infers.TensorShape
import dev.stashy.infers.internal.CloseGate
import dev.stashy.infers.internal.fromFfi
import dev.stashy.infers.internal.toFfi
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

    override fun close() {
        if (gate.markClosed()) {
            handle.close()
        }
    }
}

@OptIn(InfersInternalApi::class)
context(scope: InferenceScope)
public suspend fun GpuDeferred.materialize(): GpuImage = scope.register(
    withFfiErrors {
        ensureOpen()
        GpuImage(handle.materialize())
    },
)

@OptIn(InfersInternalApi::class)
context(scope: InferenceScope)
public suspend fun GpuPending.materialize(): GpuTensor = scope.register(
    withFfiErrors {
        ensureOpen()
        GpuTensor.fromFfi(handle.materialize())
    },
)

@OptIn(InfersInternalApi::class)
context(scope: InferenceScope)
public suspend fun GpuDeferred.process(processor: GpuImageProcessor, options: ProcessingOptions): GpuPending =
    scope.register(
        withFfiErrors {
            ensureOpen()
            processor.ensureOpen()
            GpuPending(handle.process(processor.handle, options.toFfi()))
        },
    )

@OptIn(InfersInternalApi::class)
context(scope: InferenceScope)
public suspend fun HardwareImage.on(device: GpuDevice): GpuDeferred = scope.register(
    withFfiErrors {
        ensureOpen()
        device.ensureOpen()
        GpuDeferred(handle.on(device.handle))
    },
)
