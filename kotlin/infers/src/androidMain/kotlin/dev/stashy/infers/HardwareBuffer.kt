package dev.stashy.infers

import dev.stashy.infers.internal.toFfi
import dev.stashy.infers.internal.withFfiErrors

/**
 * Thin wrapper around an Android [AHardwareBuffer][android.hardware.HardwareBuffer].
 */
public class HardwareBuffer
@InfersInternalApi
constructor(
    @property:InfersInternalApi
    public val handle: dev.stashy.infers.ffi.HardwareBufferHandle,
) : AutoCloseable {
    private val gate = dev.stashy.infers.internal.CloseGate("HardwareBuffer")

    public val width: UInt
        get() {
            gate.ensureOpen()
            return handle.width()
        }

    public val height: UInt
        get() {
            gate.ensureOpen()
            return handle.height()
        }

    public val format: UInt
        get() {
            gate.ensureOpen()
            return handle.format()
        }

    public val rawPointer: ULong
        get() {
            gate.ensureOpen()
            return handle.rawPointer()
        }

    public fun lockCpu(): ByteArray = withFfiErrors {
        gate.ensureOpen()
        handle.lockCpu()
    }

    override fun close() {
        if (gate.markClosed()) {
            handle.close()
        }
    }

    public companion object {
        @InfersInternalApi
        internal fun fromPointer(ptr: ULong, deviceInfo: DeviceInfo): HardwareBuffer = withFfiErrors {
            HardwareBuffer(
                dev.stashy.infers.ffi.createHardwareBufferFromRaw(ptr, deviceInfo.toFfi()),
            )
        }

        @InfersInternalApi
        internal fun from(buffer: android.hardware.HardwareBuffer, deviceInfo: DeviceInfo): HardwareBuffer {
            val ptr = HardwareBufferBridge.nativePointer(buffer)
            check(ptr != 0L) { "AHardwareBuffer_fromHardwareBuffer returned null" }
            return withFfiErrors {
                HardwareBuffer(
                    dev.stashy.infers.ffi.createHardwareBufferFromRaw(ptr.toULong(), deviceInfo.toFfi()),
                )
            }
        }
    }
}

internal object HardwareBufferBridge {
    init {
        try {
            System.loadLibrary("infers_bindings")
        } catch (_: UnsatisfiedLinkError) {
            // Already mapped via JNA/UniFFI.
        }
    }

    external fun nativePointer(buffer: android.hardware.HardwareBuffer): Long

    external fun nativeRelease(ptr: Long)
}
