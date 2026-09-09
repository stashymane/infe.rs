package dev.stashy.infers.vulkan

import dev.stashy.infers.Deferred
import dev.stashy.infers.FrameBuffer
import dev.stashy.infers.InfersInternalApi
import dev.stashy.infers.importedHardwareBufferOrNull
import dev.stashy.infers.internal.withFfiErrors

@OptIn(InfersInternalApi::class)
internal actual fun GpuDevice.deferFrameInternalImpl(frame: FrameBuffer<GpuDevice>): Deferred<GpuDevice> {
    ensureOpen()
    val imported = frame.importedHardwareBufferOrNull()
    if (imported != null) {
        return withFfiErrors { GpuDeferred(imported.handle.borrowOn(handle)) }
    }
    return deferHardwareInternal(frame.host)
}
