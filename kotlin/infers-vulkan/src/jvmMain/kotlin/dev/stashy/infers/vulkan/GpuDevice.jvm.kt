package dev.stashy.infers.vulkan

import dev.stashy.infers.Deferred
import dev.stashy.infers.FrameBuffer
import dev.stashy.infers.InfersInternalApi

@OptIn(InfersInternalApi::class)
internal actual fun GpuDevice.deferFrameInternalImpl(frame: FrameBuffer<GpuDevice>): Deferred<GpuDevice> =
    deferHardwareInternal(frame.host)
