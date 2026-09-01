package dev.stashy.infers.vulkan

import dev.stashy.infers.*

@OptIn(InfersInternalApi::class)
context(scope: InferenceScope)
public suspend fun GpuDevice.uploadTensor(tensor: CpuTensor): GpuTensor = scope.register(uploadTensorInternal(tensor))

@OptIn(InfersInternalApi::class)
context(scope: InferenceScope)
public suspend fun GpuDevice.uploadImage(image: DeviceImage<CpuDevice>): GpuImage =
    scope.register(uploadImageInternal(image as HostImage))

context(scope: InferenceScope)
public suspend fun CpuTensor.uploadTo(device: GpuDevice): GpuTensor = device.uploadTensor(this)

context(scope: InferenceScope)
public suspend fun DeviceImage<CpuDevice>.uploadTo(device: GpuDevice): GpuImage = device.uploadImage(this)
