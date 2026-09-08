package dev.stashy.infers.internal

import dev.stashy.infers.DataType
import dev.stashy.infers.DeviceInfo
import dev.stashy.infers.DeviceKind
import dev.stashy.infers.FitMode
import dev.stashy.infers.ImageFormat
import dev.stashy.infers.InfersException
import dev.stashy.infers.InfersInternalApi
import dev.stashy.infers.ProcessingOptions
import dev.stashy.infers.TensorLayout
import dev.stashy.infers.TensorShape
import dev.stashy.infers.ffi.DataType as FfiDataType
import dev.stashy.infers.ffi.DeviceInfo as FfiDeviceInfo
import dev.stashy.infers.ffi.DeviceKind as FfiDeviceKind
import dev.stashy.infers.ffi.FitMode as FfiFitMode
import dev.stashy.infers.ffi.ImageFormat as FfiImageFormat
import dev.stashy.infers.ffi.InfersException as FfiInfersException
import dev.stashy.infers.ffi.ProcessingOptions as FfiProcessingOptions
import dev.stashy.infers.ffi.TensorLayout as FfiTensorLayout
import dev.stashy.infers.ffi.TensorShape as FfiTensorShape

@InfersInternalApi
public fun DeviceInfo.toFfi(): FfiDeviceInfo = FfiDeviceInfo(
    kind = kind.toFfi(),
    id = id,
    name = name,
)

@InfersInternalApi
public fun FfiDeviceInfo.fromFfi(): DeviceInfo = DeviceInfo(
    kind = kind.fromFfi(),
    id = id,
    name = name,
)

@InfersInternalApi
public fun DeviceKind.toFfi(): FfiDeviceKind = when (this) {
    DeviceKind.Cpu -> FfiDeviceKind.CPU
    DeviceKind.Gpu -> FfiDeviceKind.GPU
    DeviceKind.Npu -> FfiDeviceKind.NPU
}

@InfersInternalApi
public fun FfiDeviceKind.fromFfi(): DeviceKind = when (this) {
    FfiDeviceKind.CPU -> DeviceKind.Cpu
    FfiDeviceKind.GPU -> DeviceKind.Gpu
    FfiDeviceKind.NPU -> DeviceKind.Npu
}

@InfersInternalApi
public fun FfiDataType.fromFfi(): DataType = when (this) {
    FfiDataType.U8 -> DataType.U8
    FfiDataType.I8 -> DataType.I8
    FfiDataType.I16 -> DataType.I16
    FfiDataType.I32 -> DataType.I32
    FfiDataType.I64 -> DataType.I64
    FfiDataType.F16 -> DataType.F16
    FfiDataType.F32 -> DataType.F32
    FfiDataType.F64 -> DataType.F64
}

@InfersInternalApi
public fun DataType.toFfi(): FfiDataType = when (this) {
    DataType.U8 -> FfiDataType.U8
    DataType.I8 -> FfiDataType.I8
    DataType.I16 -> FfiDataType.I16
    DataType.I32 -> FfiDataType.I32
    DataType.I64 -> FfiDataType.I64
    DataType.F16 -> FfiDataType.F16
    DataType.F32 -> FfiDataType.F32
    DataType.F64 -> FfiDataType.F64
}

@InfersInternalApi
public fun FfiImageFormat.fromFfi(): ImageFormat = when (this) {
    FfiImageFormat.RGB888 -> ImageFormat.Rgb888
    FfiImageFormat.RGBF32 -> ImageFormat.Rgbf32
    FfiImageFormat.NV12 -> ImageFormat.Nv12
    FfiImageFormat.I420 -> ImageFormat.I420
}

@InfersInternalApi
public fun ImageFormat.toFfi(): FfiImageFormat = when (this) {
    ImageFormat.Rgb888 -> FfiImageFormat.RGB888
    ImageFormat.Rgbf32 -> FfiImageFormat.RGBF32
    ImageFormat.Nv12 -> FfiImageFormat.NV12
    ImageFormat.I420 -> FfiImageFormat.I420
}

@InfersInternalApi
public fun FitMode.toFfi(): FfiFitMode = when (this) {
    FitMode.Stretch -> FfiFitMode.STRETCH
    FitMode.Contain -> FfiFitMode.CONTAIN
    FitMode.Crop -> FfiFitMode.CROP
}

@InfersInternalApi
public fun TensorLayout.toFfi(): FfiTensorLayout = when (this) {
    TensorLayout.Nhwc -> FfiTensorLayout.NHWC
    TensorLayout.Nchw -> FfiTensorLayout.NCHW
}

@InfersInternalApi
public fun TensorShape.toFfi(): FfiTensorShape = FfiTensorShape(dims = dims)

@InfersInternalApi
public fun FfiTensorShape.fromFfi(): TensorShape = TensorShape(dims = dims)

@InfersInternalApi
public fun ProcessingOptions.toFfi(): FfiProcessingOptions = FfiProcessingOptions(
    srcW = srcW,
    srcH = srcH,
    cropX = cropX,
    cropY = cropY,
    cropW = cropW,
    cropH = cropH,
    destW = destW,
    destH = destH,
    srcFormat = srcFormat.toFfi(),
    destFormat = destFormat.toFfi(),
    fitMode = fitMode.toFfi(),
    rotationDegrees = rotationDegrees,
    destLayout = destLayout.toFfi(),
)

@InfersInternalApi
public fun mapFfiException(error: FfiInfersException): InfersException = when (error) {
    is FfiInfersException.InvalidShape -> {
        InfersException.InvalidShape(error.reason)
    }

    is FfiInfersException.DeviceMismatch -> {
        InfersException.DeviceMismatch(error.expected, error.actual)
    }

    is FfiInfersException.UnsupportedType -> {
        InfersException.UnsupportedType(error.reason)
    }

    is FfiInfersException.BufferAllocationFailed -> {
        InfersException.BufferAllocationFailed(error.reason)
    }

    is FfiInfersException.ModelLoadFailed -> {
        InfersException.ModelLoadFailed(error.reason)
    }

    is FfiInfersException.InferenceFailed -> {
        InfersException.InferenceFailed(error.reason)
    }

    is FfiInfersException.ProcessingFailed -> {
        InfersException.ProcessingFailed(error.reason)
    }

    is FfiInfersException.PlatformException -> {
        InfersException.PlatformError(error.reason)
    }

    is FfiInfersException.InternalException -> {
        InfersException.InternalError(error.reason)
    }

    is FfiInfersException.AlreadyConsumed -> {
        InfersException.AlreadyConsumed()
    }
}

/** Runs [block], remapping generated UniFFI exceptions to [InfersException]. */
@InfersInternalApi public inline fun <T> withFfiErrors(block: () -> T): T = try {
    block()
} catch (e: FfiInfersException) {
    throw mapFfiException(e)
}
