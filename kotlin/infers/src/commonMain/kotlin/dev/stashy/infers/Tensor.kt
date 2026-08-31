package dev.stashy.infers

import dev.stashy.infers.ffi.createTensorFromBytes
import dev.stashy.infers.internal.CloseGate
import dev.stashy.infers.internal.TensorCodec
import dev.stashy.infers.internal.fromFfi
import dev.stashy.infers.internal.toFfi
import dev.stashy.infers.internal.withFfiErrors
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import dev.stashy.infers.ffi.TensorBuffer as FfiTensorBuffer

/**
 * Device-resident tensor. Prefer creating tensors inside [inferenceScope] so
 * they are closed automatically when the scope exits.
 *
 * Readbacks and [copyTo] are main-safe suspend functions. Native calls are not
 * interruptible — cancellation is observed between stages, not mid-readback.
 */
public class Tensor internal constructor(
    internal val handle: FfiTensorBuffer,
) : AutoCloseable {
    private val gate = CloseGate("Tensor")

    public val shape: TensorShape
        get() {
            gate.ensureOpen()
            return handle.shape().fromFfi()
        }

    public val dtype: DataType
        get() {
            gate.ensureOpen()
            return handle.dtype().fromFfi()
        }

    public val device: Device
        get() {
            gate.ensureOpen()
            return handle.device().fromFfi()
        }

    public val byteSize: ULong
        get() {
            gate.ensureOpen()
            return handle.byteSize()
        }

    /**
     * Reads tensor contents to a host [ByteArray].
     *
     * Native calls are not interruptible; cancellation is checked around the call.
     */
    public suspend fun readBytes(): ByteArray = withContext(Dispatchers.Default) {
        withFfiErrors {
            gate.ensureOpen()
            handle.readToCpuBytes()
        }
    }

    /** Decodes little-endian F32 payload after CPU readback. */
    public suspend fun readFloats(): FloatArray = TensorCodec.decodeFloats(readBytes())

    /** Decodes little-endian I32 payload after CPU readback. */
    public suspend fun readInts(): IntArray = TensorCodec.decodeInts(readBytes())

    /** Decodes little-endian I64 payload after CPU readback. */
    public suspend fun readLongs(): LongArray = TensorCodec.decodeLongs(readBytes())

    /**
     * Copies this tensor onto [device].
     *
     * Native calls are not interruptible; cancellation is checked around the call.
     */
    public suspend fun copyTo(device: Device): Tensor = withContext(Dispatchers.Default) {
        withFfiErrors {
            gate.ensureOpen()
            Tensor.fromFfi(handle.copyToDevice(device.toFfi(), null))
        }
    }

    override fun close() {
        if (gate.markClosed()) {
            handle.close()
        }
    }

    public companion object {
        internal fun fromBytes(shape: TensorShape, dtype: DataType, data: ByteArray): Tensor = withFfiErrors {
            Tensor.fromFfi(createTensorFromBytes(shape.toFfi(), dtype.toFfi(), data))
        }

        internal fun of(shape: TensorShape, data: FloatArray): Tensor =
            fromBytes(shape, DataType.F32, TensorCodec.encodeFloats(data))

        internal fun of(shape: TensorShape, data: ByteArray): Tensor = fromBytes(shape, DataType.U8, data)

        internal fun of(shape: TensorShape, data: IntArray): Tensor =
            fromBytes(shape, DataType.I32, TensorCodec.encodeInts(data))

        internal fun of(shape: TensorShape, data: LongArray): Tensor =
            fromBytes(shape, DataType.I64, TensorCodec.encodeLongs(data))
    }
}
