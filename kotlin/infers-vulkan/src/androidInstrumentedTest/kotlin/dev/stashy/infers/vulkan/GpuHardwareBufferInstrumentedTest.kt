package dev.stashy.infers.vulkan

import dev.stashy.infers.*
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals
import android.hardware.HardwareBuffer as AndroidHardwareBuffer

class GpuHardwareBufferInstrumentedTest {
    @Test
    fun gpuProcessHardwareBuffer() = runTest {
        val context = runCatching { GpuContext(Device.gpu(0u)) }.getOrNull() ?: return@runTest

        val options = ProcessingOptions {
            source = 4 to 4
            dest = 2 to 2
            srcFormat = ImageFormat.Rgb888
            destFormat = ImageFormat.Rgbf32
        }
        
        createGpuReadableRgbBuffer().use { androidBuffer ->
            context.use {
                GpuImageProcessor(it).use { processor ->
                    inferenceScope {
                        val buffer = androidBuffer.toHardwareBuffer(Device.gpu(0u))
                        val tensor = processor.process(buffer, options)
                        assertEquals(TensorShape.of(1, 2, 2, 3), tensor.shape)
                    }
                }
            }
        }
    }

    private fun createGpuReadableRgbBuffer(): AndroidHardwareBuffer = AndroidHardwareBuffer.create(
        4,
        4,
        AndroidHardwareBuffer.RGB_888,
        1,
        AndroidHardwareBuffer.USAGE_CPU_READ_OFTEN or
            AndroidHardwareBuffer.USAGE_CPU_WRITE_OFTEN or
            AndroidHardwareBuffer.USAGE_GPU_SAMPLED_IMAGE,
    )
}
