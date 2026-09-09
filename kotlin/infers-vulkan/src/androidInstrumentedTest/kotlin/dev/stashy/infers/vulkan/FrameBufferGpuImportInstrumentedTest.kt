package dev.stashy.infers.vulkan

import dev.stashy.infers.*
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals
import android.hardware.HardwareBuffer as AndroidHardwareBuffer

class FrameBufferGpuImportInstrumentedTest {
    private class GpuBuffers(
        device: GpuDevice,
    ) : BufferSet() {
        val frame by buffer(device, 4u, 4u, ImageFormat.Rgb888)
    }

    @Test
    fun importThenGpuProcessReusesSlot() = runTest {
        val device = runCatching { GpuDevice(0u) }.getOrNull() ?: return@runTest

        val options =
            ProcessingOptions {
                source = 4 to 4
                dest = 2 to 2
                srcFormat = ImageFormat.Rgb888
                destFormat = ImageFormat.Rgbf32
            }

        device.use {
            GpuImageProcessor(it).use { processor ->
                BufferPool(capacity = 1) { GpuBuffers(it) }.use { pool ->
                    createGpuReadableRgbBuffer().use { first ->
                        createGpuReadableRgbBuffer().use { second ->
                            inferenceScope {
                                val buffers = pool.checkout()
                                buffers.frame.import(first, it.info)
                                val pending = buffers.frame.on().process(processor, options)
                                assertEquals(TensorShape.of(1, 3, 2, 2), pending.shape)
                            }
                            inferenceScope {
                                val buffers = pool.checkout()
                                buffers.frame.import(second, it.info)
                                val pending = buffers.frame.on().process(processor, options)
                                assertEquals(TensorShape.of(1, 3, 2, 2), pending.shape)
                            }
                        }
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
