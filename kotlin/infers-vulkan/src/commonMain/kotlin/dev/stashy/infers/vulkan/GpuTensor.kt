package dev.stashy.infers.vulkan

import dev.stashy.infers.ByteTensorView
import dev.stashy.infers.CpuTensor
import dev.stashy.infers.DeviceInfo
import dev.stashy.infers.FloatTensorView
import dev.stashy.infers.InfersInternalApi
import dev.stashy.infers.IntTensorView
import dev.stashy.infers.LongTensorView
import dev.stashy.infers.Tensor
import dev.stashy.infers.TensorShape
import dev.stashy.infers.internal.CloseGate
import dev.stashy.infers.internal.createByteTensorView
import dev.stashy.infers.internal.createFloatTensorView
import dev.stashy.infers.internal.createIntTensorView
import dev.stashy.infers.internal.createLongTensorView
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

    override suspend fun floats(): FloatTensorView = withContext(Dispatchers.Default) {
        val cpu = download()
        try {
            createFloatTensorView(cpu.handle, cpu, closeKeepAlive = true)
        } catch (t: Throwable) {
            cpu.close()
            throw t
        }
    }

    override suspend fun ints(): IntTensorView = withContext(Dispatchers.Default) {
        val cpu = download()
        try {
            createIntTensorView(cpu.handle, cpu, closeKeepAlive = true)
        } catch (t: Throwable) {
            cpu.close()
            throw t
        }
    }

    override suspend fun longs(): LongTensorView = withContext(Dispatchers.Default) {
        val cpu = download()
        try {
            createLongTensorView(cpu.handle, cpu, closeKeepAlive = true)
        } catch (t: Throwable) {
            cpu.close()
            throw t
        }
    }

    override suspend fun bytes(): ByteTensorView = withContext(Dispatchers.Default) {
        val cpu = download()
        try {
            createByteTensorView(cpu.handle, cpu, closeKeepAlive = true)
        } catch (t: Throwable) {
            cpu.close()
            throw t
        }
    }

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
