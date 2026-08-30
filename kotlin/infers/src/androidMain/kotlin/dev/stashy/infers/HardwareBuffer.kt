package dev.stashy.infers

import dev.stashy.infers.ffi.HardwareBufferHandle
import dev.stashy.infers.ffi.createHardwareBufferFromRaw
import dev.stashy.infers.internal.CloseGate
import dev.stashy.infers.internal.toFfi
import dev.stashy.infers.internal.withFfiErrors

/**
 * Thin wrapper around an Android [AHardwareBuffer][android.hardware.HardwareBuffer].
 *
 * Prefer creating instances via [android.hardware.HardwareBuffer.toHardwareBuffer]
 * inside [inferenceScope] so the buffer is closed when the scope exits.
 * Use [fromPointer] only when you already have a raw `AHardwareBuffer*` from NDK code.
 */
public class HardwareBuffer
@InfersInternalApi
constructor(
    @property:InfersInternalApi
    public val handle: HardwareBufferHandle,
) : AutoCloseable {
    private val gate = CloseGate("HardwareBuffer")

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
        public fun fromPointer(ptr: ULong, device: Device): HardwareBuffer = withFfiErrors {
            HardwareBuffer(createHardwareBufferFromRaw(ptr, device.toFfi()))
        }

        /**
         * Wrap a platform [android.hardware.HardwareBuffer], converting it to an
         * `AHardwareBuffer*` via the NDK JNI bridge.
         */
        public fun from(buffer: android.hardware.HardwareBuffer, device: Device): HardwareBuffer {
            val ptr = HardwareBufferBridge.nativePointer(buffer)
            check(ptr != 0L) { "AHardwareBuffer_fromHardwareBuffer returned null" }
            // Rust `createHardwareBufferFromRaw` acquires its own +1 and releases
            // on Drop. Do not call nativeRelease here — that would over-free the
            // GraphicBuffer still owned by [buffer] and crash on close (MTE).
            return withFfiErrors {
                HardwareBuffer(createHardwareBufferFromRaw(ptr.toULong(), device.toFfi()))
            }
        }
    }
}

/** JNI entry points implemented in `libinfers_bindings` (`android_jni.rs`). */
internal object HardwareBufferBridge {
    init {
        try {
            System.loadLibrary("infers_bindings")
        } catch (_: UnsatisfiedLinkError) {
            // Already mapped via JNA/UniFFI.
        }
    }

    external fun nativePointer(buffer: android.hardware.HardwareBuffer): Long

    /** Drops one `AHardwareBuffer` ref; not used by [HardwareBuffer.from] (ownership is transferred). */
    external fun nativeRelease(ptr: Long)
}
