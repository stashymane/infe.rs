package dev.stashy.infers

/**
 * Wrap a platform HardwareBuffer; owned by the enclosing [inferenceScope]:
 * `cameraBuffer.toHardwareBuffer(gpuDevice.info)`.
 */
@OptIn(InfersInternalApi::class)
context(scope: InferenceScope)
public fun android.hardware.HardwareBuffer.toHardwareBuffer(deviceInfo: DeviceInfo): HardwareBuffer =
    scope.register(HardwareBuffer.from(this, deviceInfo))
