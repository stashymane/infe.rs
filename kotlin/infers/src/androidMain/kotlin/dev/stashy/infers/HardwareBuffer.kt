package dev.stashy.infers

import dev.stashy.infers.internal.toFfi
import dev.stashy.infers.internal.withFfiErrors

/**
 * Thin wrapper around an Android [AHardwareBuffer][android.hardware.HardwareBuffer].
 *
 * Owns an independent native +1 acquired when wrapping a Java
 * [android.hardware.HardwareBuffer] (NDK `fromHardwareBuffer` does not acquire).
 * The Java object remains caller-owned (e.g. camera `Image.close()`).
 * Calling [dev.stashy.infers.vulkan.on] transfers Rust ownership into deferred
 * Vulkan import and closes this wrapper.
 */
public class HardwareBuffer
@InfersInternalApi
constructor(
    @property:InfersInternalApi
    public val handle: dev.stashy.infers.ffi.HardwareBufferHandle,
) : AutoCloseable {
    private val gate = dev.stashy.infers.internal.CloseGate("HardwareBuffer")

    public val width: UInt
        get() = withFfiErrors {
            gate.ensureOpen()
            handle.width()
        }

    public val height: UInt
        get() = withFfiErrors {
            gate.ensureOpen()
            handle.height()
        }

    public val format: UInt
        get() = withFfiErrors {
            gate.ensureOpen()
            handle.format()
        }

    public val rawPointer: ULong
        get() = withFfiErrors {
            gate.ensureOpen()
            handle.rawPointer()
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
                // Caller transfers an existing +1 (e.g. from allocate).
                dev.stashy.infers.ffi.createHardwareBufferFromOwned(ptr, deviceInfo.toFfi()),
            )
        }

        @InfersInternalApi
        internal fun from(buffer: android.hardware.HardwareBuffer, deviceInfo: DeviceInfo): HardwareBuffer {
            val ptr = HardwareBufferBridge.nativePointer(buffer)
            check(ptr != 0L) { "AHardwareBuffer_fromHardwareBuffer returned null" }
            // Borrowed pointer: acquire an independent +1 inside from_java.
            return withFfiErrors {
                HardwareBuffer(
                    dev.stashy.infers.ffi.createHardwareBufferFromJava(ptr.toULong(), deviceInfo.toFfi()),
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
}
