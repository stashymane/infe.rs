package dev.stashy.infers

import dev.stashy.infers.internal.CloseGate
import dev.stashy.infers.internal.fromFfi
import dev.stashy.infers.ffi.CpuImage as FfiCpuImage

/** CPU-resident image input (zero-copy wrap of [HardwareImage] bytes). */
public class CpuImage internal constructor(
    internal val handle: FfiCpuImage,
) : DeviceImage<CpuDevice> {
    private val gate = CloseGate("CpuImage")

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
