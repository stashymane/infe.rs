package dev.stashy.infers.vulkan

import dev.stashy.infers.DeviceImage
import dev.stashy.infers.ImageFormat
import dev.stashy.infers.internal.CloseGate
import dev.stashy.infers.internal.fromFfi
import dev.stashy.infers.ffi.GpuImage as FfiGpuImage

/** GPU-resident image input. */
public class GpuImage internal constructor(
    internal val handle: FfiGpuImage,
) : DeviceImage<GpuDevice> {
    private val gate = CloseGate("GpuImage")

    override val width: UInt
        get() {
            gate.ensureOpen()
            return handle.width()
        }

    override val height: UInt
        get() {
            gate.ensureOpen()
            return handle.height()
        }

    override val format: ImageFormat
        get() {
            gate.ensureOpen()
            return handle.format().fromFfi()
        }

    override fun close() {
        if (gate.markClosed()) {
            handle.close()
        }
    }
}
