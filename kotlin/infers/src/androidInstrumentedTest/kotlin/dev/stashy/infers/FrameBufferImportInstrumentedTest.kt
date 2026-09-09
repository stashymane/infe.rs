package dev.stashy.infers

import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertSame
import android.hardware.HardwareBuffer as AndroidHardwareBuffer

@OptIn(InfersInternalApi::class)
class FrameBufferImportInstrumentedTest {
    private class CameraBuffers : BufferSet() {
        val frame by buffer(CpuDevice, 4u, 4u, ImageFormat.Rgb888)
    }

    @Test
    fun importIntoExistingSlotThenCpuProcess() = runTest {
        val options =
            ProcessingOptions {
                source = 4 to 4
                dest = 2 to 2
                srcFormat = ImageFormat.Rgb888
                destFormat = ImageFormat.Rgbf32
            }

        BufferPool(capacity = 1) { CameraBuffers() }.use { pool ->
            CpuImageProcessor().use { processor ->
                createRgbBuffer().use { androidBuffer ->
                    inferenceScope {
                        val buffers = pool.checkout()
                        buffers.frame.import(androidBuffer, CpuDevice.info)
                        assertNotNull(buffers.frame.importedHardwareBufferOrNull())
                        val pending = buffers.frame.on().process(processor, options)
                        assertEquals(TensorShape.of(1, 3, 2, 2), pending.shape)
                    }
                    // Lease return clears the import; Java buffer remains usable.
                    assertEquals(4, androidBuffer.width)
                }
            }
        }
    }

    @Test
    fun reimportIntoSameSlotAfterLease() = runTest {
        BufferPool(capacity = 1) { CameraBuffers() }.use { pool ->
            createRgbBuffer().use { first ->
                createRgbBuffer().use { second ->
                    val slot =
                        inferenceScope {
                            val buffers = pool.checkout()
                            buffers.frame.import(first, CpuDevice.info)
                            buffers.frame
                        }
                    assertNull(slot.importedHardwareBufferOrNull())
                    inferenceScope {
                        val buffers = pool.checkout()
                        assertSame(slot, buffers.frame)
                        buffers.frame.import(second, CpuDevice.info)
                        assertNotNull(buffers.frame.importedHardwareBufferOrNull())
                    }
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
