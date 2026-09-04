package dev.stashy.infers.vulkan

import dev.stashy.infers.HardwareBuffer
import dev.stashy.infers.InferenceScope
import dev.stashy.infers.InfersInternalApi
import dev.stashy.infers.internal.withFfiErrors

/** Defer zero-copy Vulkan import of [buffer] onto [device]. */
@OptIn(InfersInternalApi::class)
context(scope: InferenceScope)
public fun HardwareBuffer.on(device: GpuDevice): GpuDeferred = scope.register(
    withFfiErrors {
        device.ensureOpen()
        GpuDeferred(handle.on(device.handle))
    },
)
