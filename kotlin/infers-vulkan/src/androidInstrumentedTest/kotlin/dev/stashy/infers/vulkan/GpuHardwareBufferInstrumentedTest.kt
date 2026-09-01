package dev.stashy.infers.vulkan

import dev.stashy.infers.*
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals
import android.hardware.HardwareBuffer as AndroidHardwareBuffer

class GpuHardwareBufferInstrumentedTest {
    @Test
    fun gpuProcessHardwareBuffer() = runTest {
        val device = runCatching { GpuDevice(0u) }.getOrNull() ?: return@runTest

        val options = ProcessingOptions {
            source = 4 to 4
            dest = 2 to 2
            srcFormat = ImageFormat.Rgb888
            destFormat = ImageFormat.Rgbf32
        }

        createGpuReadableRgbBuffer().use { androidBuffer ->
            device.use {
                GpuImageProcessor(it).use { processor ->
                    inferenceScope {
                        val buffer = androidBuffer.toHardwareBuffer(it.info)
                        val tensor = processor.processHardwareBuffer(buffer, options)
                        assertEquals(TensorShape.of(1, 3, 2, 2), tensor.shape)
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
