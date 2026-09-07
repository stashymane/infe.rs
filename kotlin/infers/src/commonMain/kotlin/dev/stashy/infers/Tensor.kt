package dev.stashy.infers

import dev.stashy.infers.ffi.createCpuTensorFromBytes
import dev.stashy.infers.internal.CloseGate
import dev.stashy.infers.internal.TensorCodec
import dev.stashy.infers.internal.createByteTensorView
import dev.stashy.infers.internal.createFloatTensorView
import dev.stashy.infers.internal.createIntTensorView
import dev.stashy.infers.internal.createLongTensorView
import dev.stashy.infers.internal.fromFfi
import dev.stashy.infers.internal.toFfi
import dev.stashy.infers.internal.withFfiErrors
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import dev.stashy.infers.ffi.CpuTensor as FfiCpuTensor

/**
 * Device-resident tensor on execution device [D]. Prefer creating tensors inside
 * [inferenceScope] so they are closed automatically when the scope exits.
 *
 * Read results via [floats] / [ints] / [longs] / [bytes] views. Native calls are not
 * interruptible — cancellation is observed between stages, not mid-readback.
 */
public interface Tensor<D : Device> : AutoCloseable {
    public val shape: TensorShape
    public val dtype: DataType
    public val deviceInfo: DeviceInfo
    public val byteSize: ULong

    /** Typed float view over host storage (no full-tensor allocation). */
    public suspend fun floats(): FloatTensorView

    /** Typed int view over host storage (no full-tensor allocation). */
    public suspend fun ints(): IntTensorView

    /** Typed long view over host storage (no full-tensor allocation). */
    public suspend fun longs(): LongTensorView

    /** Byte view over host storage (no full-tensor allocation). */
    public suspend fun bytes(): ByteTensorView
}

/** Host-resident CPU tensor. */
public class CpuTensor @InfersInternalApi constructor(
    @InfersInternalApi
    public val handle: FfiCpuTensor,
) : Tensor<CpuDevice> {
    private val gate = CloseGate("CpuTensor")

    override val shape: TensorShape
        get() {
            gate.ensureOpen()
            return handle.shape().fromFfi()
        }

    override val dtype: DataType
        get() {
            gate.ensureOpen()
            return handle.dtype().fromFfi()
        }

    override val deviceInfo: DeviceInfo = CpuDevice.info

    override val byteSize: ULong
        get() {
            gate.ensureOpen()
            return handle.byteSize()
        }

    override suspend fun floats(): FloatTensorView = withContext(Dispatchers.Default) {
        withFfiErrors {
            gate.ensureOpen()
            createFloatTensorView(handle, this@CpuTensor)
        }
    }

    override suspend fun ints(): IntTensorView = withContext(Dispatchers.Default) {
        withFfiErrors {
            gate.ensureOpen()
            createIntTensorView(handle, this@CpuTensor)
        }
    }

    override suspend fun longs(): LongTensorView = withContext(Dispatchers.Default) {
        withFfiErrors {
            gate.ensureOpen()
            createLongTensorView(handle, this@CpuTensor)
        }
    }

    override suspend fun bytes(): ByteTensorView = withContext(Dispatchers.Default) {
        withFfiErrors {
            gate.ensureOpen()
            createByteTensorView(handle, this@CpuTensor)
        }
    }

    @InfersInternalApi
    public fun ensureOpen(): Unit = gate.ensureOpen()

    override fun close() {
        if (gate.markClosed()) {
            handle.close()
        }
    }

    public companion object {
        @InfersInternalApi
        public fun fromFfi(handle: FfiCpuTensor): CpuTensor = CpuTensor(handle)

        internal fun fromBytes(shape: TensorShape, dtype: DataType, data: ByteArray): CpuTensor = withFfiErrors {
            fromFfi(createCpuTensorFromBytes(shape.toFfi(), dtype.toFfi(), data))
        }

        internal fun of(shape: TensorShape, data: FloatArray): CpuTensor =
            fromBytes(shape, DataType.F32, TensorCodec.encodeFloats(data))

        internal fun of(shape: TensorShape, data: ByteArray): CpuTensor = fromBytes(shape, DataType.U8, data)

        internal fun of(shape: TensorShape, data: IntArray): CpuTensor =
            fromBytes(shape, DataType.I32, TensorCodec.encodeInts(data))

        internal fun of(shape: TensorShape, data: LongArray): CpuTensor =
            fromBytes(shape, DataType.I64, TensorCodec.encodeLongs(data))
    }
}
