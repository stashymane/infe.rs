package dev.stashy.infers.internal

import dev.stashy.infers.ByteTensorView
import dev.stashy.infers.FloatTensorView
import dev.stashy.infers.InfersInternalApi
import dev.stashy.infers.IntTensorView
import dev.stashy.infers.LongTensorView
import dev.stashy.infers.ffi.CpuTensor as FfiCpuTensor

@InfersInternalApi
public actual fun createFloatTensorView(
    handle: FfiCpuTensor,
    keepAlive: AutoCloseable,
    closeKeepAlive: Boolean,
): FloatTensorView = DirectHostViews.floats(handle, keepAlive, closeKeepAlive)

@InfersInternalApi
public actual fun createIntTensorView(
    handle: FfiCpuTensor,
    keepAlive: AutoCloseable,
    closeKeepAlive: Boolean,
): IntTensorView = DirectHostViews.ints(handle, keepAlive, closeKeepAlive)

@InfersInternalApi
public actual fun createLongTensorView(
    handle: FfiCpuTensor,
    keepAlive: AutoCloseable,
    closeKeepAlive: Boolean,
): LongTensorView = DirectHostViews.longs(handle, keepAlive, closeKeepAlive)

@InfersInternalApi
public actual fun createByteTensorView(
    handle: FfiCpuTensor,
    keepAlive: AutoCloseable,
    closeKeepAlive: Boolean,
): ByteTensorView = DirectHostViews.bytes(handle, keepAlive, closeKeepAlive)
