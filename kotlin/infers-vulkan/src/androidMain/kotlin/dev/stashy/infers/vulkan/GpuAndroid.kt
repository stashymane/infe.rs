package dev.stashy.infers.vulkan

import dev.stashy.infers.HardwareBuffer
import dev.stashy.infers.InferenceScope
import dev.stashy.infers.InfersInternalApi
import dev.stashy.infers.internal.withFfiErrors

/**
 * Defer zero-copy Vulkan import of this buffer onto [device].
 *
 * Consumes the native AHB ownership held by this [HardwareBuffer] and closes
 * the wrapper so the buffer cannot be reused after scheduling import.
 */
@OptIn(InfersInternalApi::class)
context(scope: InferenceScope)
public fun HardwareBuffer.on(device: GpuDevice): GpuDeferred = scope.register(
    withFfiErrors {
        device.ensureOpen()
        val deferred = GpuDeferred(handle.on(device.handle))
        // Ownership moved into Deferred; invalidate this wrapper for scope cleanup.
        close()
        deferred
    },
)
