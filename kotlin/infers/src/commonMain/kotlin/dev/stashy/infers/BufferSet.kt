package dev.stashy.infers

import kotlin.properties.ReadOnlyProperty

/**
 * User-defined named frame slots for [BufferPool].
 *
 * Subclass and declare properties with [buffer]:
 * ```
 * class CameraBuffers(gpu: GpuDevice, w: UInt, h: UInt) : BufferSet() {
 *     val frame by buffer(gpu, w, h, ImageFormat.Rgb888)
 * }
 * ```
 */
public open class BufferSet {
    private val slots = ArrayDeque<FrameBuffer<*>>()
    private var sealed: Boolean = false

    /**
     * Declares a reusable [FrameBuffer] slot bound to [device] during construction.
     *
     * Storage is host-resident capacity of [width]×[height]×[format]; later
     * [FrameBuffer.on] targets [device].
     */
    protected fun <D : Device<D>> buffer(
        device: D,
        width: UInt,
        height: UInt,
        format: ImageFormat,
    ): ReadOnlyProperty<BufferSet, FrameBuffer<D>> {
        check(!sealed) { "buffer() may only be called during BufferSet construction" }
        val frame = FrameBuffer.allocate(device, width, height, format)
        slots.addLast(frame)
        return ReadOnlyProperty<BufferSet, FrameBuffer<D>> { _, _ -> frame }
    }

    internal fun seal() {
        sealed = true
    }

    internal fun clearImports() {
        for (slot in slots) {
            slot.clearImport()
        }
    }

    internal fun destroySlots() {
        var primary: Throwable? = null
        while (slots.isNotEmpty()) {
            try {
                slots.removeLast().close()
            } catch (t: Throwable) {
                if (primary == null) {
                    primary = t
                } else {
                    primary.addSuppressed(t)
                }
            }
        }
        if (primary != null) {
            throw primary
        }
    }
}
