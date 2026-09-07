package dev.stashy.infers

import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import android.hardware.HardwareBuffer as AndroidHardwareBuffer

class HardwareBufferInstrumentedTest {
    @Test
    fun wrapPlatformHardwareBuffer() = runTest {
        createRgbBuffer().use { androidBuffer ->
            inferenceScope {
                val buffer = androidBuffer.toHardwareBuffer(CpuDevice.info)
                assertEquals(4u, buffer.width)
                assertEquals(4u, buffer.height)
            }
        }
    }

    @Test
    fun cpuProcessHardwareBuffer() = runTest {
        val options = ProcessingOptions {
            source = 4 to 4
            dest = 2 to 2
            srcFormat = ImageFormat.Rgb888
            destFormat = ImageFormat.Rgbf32
        }

        createRgbBuffer().use { androidBuffer ->
            CpuImageProcessor().use { processor ->
                inferenceScope {
                    val buffer = androidBuffer.toHardwareBuffer(CpuDevice.info)
                    val bytes = buffer.lockCpu()
                    val hardware = HardwareImage.fromBytes(
                        bytes,
                        buffer.width,
                        buffer.height,
                        ImageFormat.Rgb888,
                    )
                    val pending = hardware.onCpu().process(processor, options)
                    assertEquals(TensorShape.of(1, 3, 2, 2), pending.shape)
                    assertTrue(pending.materialize().floats().use { it.size > 0 })
                }
            }
        }
    }

    private fun createRgbBuffer(): AndroidHardwareBuffer = AndroidHardwareBuffer.create(
        4,
        4,
        AndroidHardwareBuffer.RGB_888,
        1,
        AndroidHardwareBuffer.USAGE_CPU_READ_OFTEN or
            AndroidHardwareBuffer.USAGE_CPU_WRITE_OFTEN,
    )
}
