package dev.stashy.infers.vulkan

import dev.stashy.infers.HardwareBuffer
import dev.stashy.infers.InferenceScope
import dev.stashy.infers.InfersInternalApi
import dev.stashy.infers.ProcessingOptions
import dev.stashy.infers.internal.toFfi
import dev.stashy.infers.internal.withFfiErrors
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

@OptIn(InfersInternalApi::class)
context(scope: InferenceScope)
public suspend fun GpuImageProcessor.processHardwareBuffer(
    buffer: HardwareBuffer,
    options: ProcessingOptions,
): GpuTensor = withContext(Dispatchers.Default) {
    withFfiErrors {
        ensureOpen()
        scope.register(
            GpuTensor.fromFfi(
                handle.processHardwareBuffer(buffer.handle, options.toFfi()),
            ),
        )
    }
}
