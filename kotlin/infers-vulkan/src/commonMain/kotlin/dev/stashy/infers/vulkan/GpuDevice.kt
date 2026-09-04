package dev.stashy.infers.vulkan

import dev.stashy.infers.CpuTensor
import dev.stashy.infers.Device
import dev.stashy.infers.DeviceInfo
import dev.stashy.infers.InfersInternalApi
import dev.stashy.infers.internal.CloseGate
import dev.stashy.infers.internal.fromFfi
import dev.stashy.infers.internal.withFfiErrors
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import dev.stashy.infers.ffi.GpuDevice as FfiGpuDevice

/** GPU execution device. Owns the shared Vulkan context. */
@OptIn(InfersInternalApi::class)
public class GpuDevice private constructor(
    internal val handle: FfiGpuDevice,
) : Device,
    AutoCloseable {
    private val gate = CloseGate("GpuDevice")

    public constructor(id: ULong = 0u) : this(
        withFfiErrors { FfiGpuDevice(id) },
    )

    override val info: DeviceInfo
        get() {
            gate.ensureOpen()
            return handle.info().fromFfi()
        }

    @InfersInternalApi
    internal fun ensureOpen() = gate.ensureOpen()

    @InfersInternalApi
    internal suspend fun uploadTensorInternal(tensor: CpuTensor): GpuTensor = withContext(Dispatchers.Default) {
        withFfiErrors {
            gate.ensureOpen()
            tensor.ensureOpen()
            GpuTensor.fromFfi(handle.uploadTensor(tensor.handle))
        }
    }

    override fun close() {
        if (gate.markClosed()) {
            handle.close()
        }
    }
}
