package dev.stashy.infers

import dev.stashy.infers.ffi.createCpuTensorFromBytes
import dev.stashy.infers.internal.CloseGate
import dev.stashy.infers.internal.TensorCodec
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
 * Readbacks are main-safe suspend functions. Native calls are not interruptible —
 * cancellation is observed between stages, not mid-readback.
 */
public interface Tensor<D : Device> : AutoCloseable {
    public val shape: TensorShape
    public val dtype: DataType
    public val deviceInfo: DeviceInfo
    public val byteSize: ULong

    public suspend fun readBytes(): ByteArray

    public suspend fun readFloats(): FloatArray

    public suspend fun readInts(): IntArray

    public suspend fun readLongs(): LongArray
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

    override suspend fun readBytes(): ByteArray = withContext(Dispatchers.Default) {
        withFfiErrors {
            gate.ensureOpen()
            handle.readBytes()
        }
    }

    override suspend fun readFloats(): FloatArray = TensorCodec.decodeFloats(readBytes())

    override suspend fun readInts(): IntArray = TensorCodec.decodeInts(readBytes())

    override suspend fun readLongs(): LongArray = TensorCodec.decodeLongs(readBytes())

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
