package dev.stashy.infers.internal

import dev.stashy.infers.ByteTensorView
import dev.stashy.infers.FloatTensorView
import dev.stashy.infers.InfersInternalApi
import dev.stashy.infers.IntTensorView
import dev.stashy.infers.LongTensorView
import dev.stashy.infers.ffi.CpuTensor as FfiCpuTensor

@InfersInternalApi
public expect fun createFloatTensorView(
    handle: FfiCpuTensor,
    keepAlive: AutoCloseable,
    closeKeepAlive: Boolean = false,
): FloatTensorView

@InfersInternalApi
public expect fun createIntTensorView(
    handle: FfiCpuTensor,
    keepAlive: AutoCloseable,
    closeKeepAlive: Boolean = false,
): IntTensorView

@InfersInternalApi
public expect fun createLongTensorView(
    handle: FfiCpuTensor,
    keepAlive: AutoCloseable,
    closeKeepAlive: Boolean = false,
): LongTensorView

@InfersInternalApi
public expect fun createByteTensorView(
    handle: FfiCpuTensor,
    keepAlive: AutoCloseable,
    closeKeepAlive: Boolean = false,
): ByteTensorView
