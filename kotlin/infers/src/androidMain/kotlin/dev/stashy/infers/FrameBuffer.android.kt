package dev.stashy.infers

/**
 * Bind an Android hardware buffer into this existing [FrameBuffer] slot.
 *
 * Acquires an independent native +1; the Java [android.hardware.HardwareBuffer]
 * (and camera [android.media.Image]) remain caller-owned. Replaces any previous
 * import. Cleared when the pool lease returns or [FrameBuffer.write] is called.
 */
@OptIn(InfersInternalApi::class)
public fun <D : Device<D>> FrameBuffer<D>.import(buffer: HardwareBuffer) {
    require(buffer.width == width && buffer.height == height) {
        "HardwareBuffer size ${buffer.width}x${buffer.height} does not match slot ${width}x$height"
    }
    // Move ownership of this wrapper into the slot (caller should not close it).
    platformImport.install(buffer)
}

/**
 * Import a platform [android.hardware.HardwareBuffer] into this slot
 * (acquires an independent +1; Java object stays caller-owned).
 */
@OptIn(InfersInternalApi::class)
public fun <D : Device<D>> FrameBuffer<D>.import(buffer: android.hardware.HardwareBuffer, deviceInfo: DeviceInfo) {
    import(HardwareBuffer.from(buffer, deviceInfo))
}

/** Currently imported buffer, if any (for GPU borrow-on). */
@InfersInternalApi
public fun <D : Device<D>> FrameBuffer<D>.importedHardwareBufferOrNull(): HardwareBuffer? = platformImport.peek()
