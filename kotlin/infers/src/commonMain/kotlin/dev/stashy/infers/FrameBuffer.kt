package dev.stashy.infers

import dev.stashy.infers.internal.CloseGate

/**
 * Long-lived reusable frame slot for [BufferSet] / [BufferPool], bound to device [D].
 *
 * Host capacity is allocated once. Call [write] to refill from bytes, or on Android
 * [import] a hardware buffer into this same slot. Destroyed only when the owning
 * set/pool closes — not when a pool lease returns.
 */
public class FrameBuffer<D : Device<D>>
internal constructor(
    public val device: D,
    @property:InfersInternalApi
    public val host: HardwareImage,
    internal val platformImport: PlatformFrameImport = PlatformFrameImport(),
) : AutoCloseable {
    private val gate = CloseGate("FrameBuffer")

    public val width: UInt
        get() {
            gate.ensureOpen()
            return host.width
        }

    public val height: UInt
        get() {
            gate.ensureOpen()
            return host.height
        }

    public val format: ImageFormat
        get() {
            gate.ensureOpen()
            return host.format
        }

    /** Overwrite host pixels. Clears any imported hardware buffer. */
    public fun write(bytes: ByteArray) {
        gate.ensureOpen()
        clearImport()
        host.write(bytes)
    }

    /** Release any imported hardware buffer; host capacity is kept. */
    public fun clearImport() {
        platformImport.clear()
    }

    internal fun prepareHostForDefer() {
        platformImport.copyIntoHostIfNeeded(host)
    }

    override fun close() {
        if (gate.markClosed()) {
            clearImport()
            host.close()
        }
    }

    public companion object {
        @OptIn(InfersInternalApi::class)
        internal fun <D : Device<D>> allocate(
            device: D,
            width: UInt,
            height: UInt,
            format: ImageFormat,
        ): FrameBuffer<D> = FrameBuffer(device, HardwareImage.empty(width, height, format))
    }
}

/**
 * Place this frame on its bound [FrameBuffer.device] without committing transfer/compute.
 *
 * Dispatches through [Device.deferFrameInternal] (CPU copy-from-import, GPU may
 * zero-copy an imported Android hardware buffer).
 */
@OptIn(InfersInternalApi::class)
context(scope: InferenceScope)
public fun <D : Device<D>> FrameBuffer<D>.on(): Deferred<D> = scope.register(device.deferFrameInternal(this))

/** Platform-specific imported-buffer state for a [FrameBuffer]. */
internal expect class PlatformFrameImport() {
    fun hasImport(): Boolean

    fun clear()

    fun copyIntoHostIfNeeded(host: HardwareImage)
}
