package dev.stashy.infers

import dev.stashy.infers.internal.fromFfi
import dev.stashy.infers.internal.toFfi
import dev.stashy.infers.internal.withFfiErrors
import kotlinx.coroutines.withContext

/**
 * Zero-copy Android [HardwareBuffer] preprocessing inside [inferenceScope]:
 * `processor.process(buffer, options)`.
 */
@OptIn(InfersInternalApi::class)
context(scope: InferenceScope)
public suspend fun ImageProcessor.process(
    buffer: HardwareBuffer,
    options: ProcessingOptions,
): Tensor {
    val ffiProcessor =
        this as? FfiBackedImageProcessor
            ?: error("ImageProcessor must be an Infers FFI-backed processor")
    return scope.register(ffiProcessor.processHardwareBuffer(buffer, options))
}

/**
 * Wrap a platform HardwareBuffer; owned by the enclosing [inferenceScope]:
 * `cameraBuffer.toHardwareBuffer(Device.gpu())`.
 */
@OptIn(InfersInternalApi::class)
context(scope: InferenceScope)
public fun android.hardware.HardwareBuffer.toHardwareBuffer(device: Device): HardwareBuffer =
    scope.register(HardwareBuffer.from(this, device))

@InfersInternalApi
public suspend fun FfiBackedImageProcessor.processHardwareBuffer(
    buffer: HardwareBuffer,
    options: ProcessingOptions,
): Tensor =
    withContext(dispatcher) {
        withFfiErrors {
            ensureOpen()
            Tensor.fromFfi(handle.processHardwareBuffer(buffer.handle, options.toFfi()))
        }
    }
