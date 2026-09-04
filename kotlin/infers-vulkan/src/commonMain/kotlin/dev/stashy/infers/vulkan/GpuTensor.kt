package dev.stashy.infers.vulkan

import dev.stashy.infers.CpuTensor
import dev.stashy.infers.DeviceInfo
import dev.stashy.infers.InfersInternalApi
import dev.stashy.infers.Tensor
import dev.stashy.infers.TensorShape
import dev.stashy.infers.internal.CloseGate
import dev.stashy.infers.internal.fromFfi
import dev.stashy.infers.internal.withFfiErrors
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import dev.stashy.infers.ffi.GpuTensor as FfiGpuTensor

/** GPU-resident tensor. */
public class GpuTensor internal constructor(
    internal val handle: FfiGpuTensor,
) : Tensor<GpuDevice> {
    private val gate = CloseGate("GpuTensor")

    override val shape: TensorShape
        get() {
            gate.ensureOpen()
            return handle.shape().fromFfi()
        }

    override val dtype: dev.stashy.infers.DataType
        get() {
            gate.ensureOpen()
            return handle.dtype().fromFfi()
        }

    override val deviceInfo: DeviceInfo
        get() {
            gate.ensureOpen()
            return handle.deviceInfo().fromFfi()
        }

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

    override suspend fun readFloats(): FloatArray = dev.stashy.infers.internal.TensorCodec.decodeFloats(readBytes())

    override suspend fun readInts(): IntArray = dev.stashy.infers.internal.TensorCodec.decodeInts(readBytes())

    override suspend fun readLongs(): LongArray = dev.stashy.infers.internal.TensorCodec.decodeLongs(readBytes())

    public suspend fun download(): CpuTensor = withContext(Dispatchers.Default) {
        withFfiErrors {
            gate.ensureOpen()
            CpuTensor.fromFfi(handle.download())
        }
    }

    @InfersInternalApi
    internal fun ensureOpen(): Unit = gate.ensureOpen()

    override fun close() {
        if (gate.markClosed()) {
            handle.close()
        }
    }

    internal companion object {
        fun fromFfi(handle: FfiGpuTensor): GpuTensor = GpuTensor(handle)
    }
}
