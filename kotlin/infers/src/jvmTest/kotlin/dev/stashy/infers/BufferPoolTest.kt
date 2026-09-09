package dev.stashy.infers

import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertSame

class BufferPoolTest {
    private class TestBuffers(
        width: UInt = 2u,
        height: UInt = 2u,
    ) : BufferSet() {
        val frame by buffer(CpuDevice, width, height, ImageFormat.Rgb888)
    }

    @Test
    fun checkoutReturnsSameSlotsAcrossLeases() = runTest {
        BufferPool(capacity = 1) { TestBuffers() }.use { pool ->
            val first =
                inferenceScope {
                    val buffers = pool.checkout()
                    buffers.frame.write(ByteArray(12) { it.toByte() })
                    buffers
                }
            val second =
                inferenceScope {
                    pool.checkout()
                }
            assertSame(first.frame, second.frame)
        }
    }

    @Test
    fun thirdCheckoutThrowsWhenCapacityIsTwo() = runTest {
        BufferPool(capacity = 2) { TestBuffers() }.use { pool ->
            inferenceScope {
                pool.checkout()
                pool.checkout()
                val error =
                    assertFailsWith<InfersException.BufferPoolExhausted> {
                        pool.checkout()
                    }
                assertEquals("capacity=2, all sets are in use", error.reason)
            }
        }
    }

    @Test
    fun writeThenOnCpuProcessesPooledFrame() = runTest {
        val options =
            ProcessingOptions {
                source = 2 to 2
                dest = 2 to 2
                srcFormat = ImageFormat.Rgb888
                destFormat = ImageFormat.Rgbf32
            }
        BufferPool(capacity = 1) { TestBuffers() }.use { pool ->
            CpuImageProcessor().use { processor ->
                inferenceScope {
                    val buffers = pool.checkout()
                    buffers.frame.write(ByteArray(12) { (it * 3).toByte() })
                    val pending = buffers.frame.on().process(processor, options)
                    assertEquals(TensorShape.of(1, 3, 2, 2), pending.shape)
                }
            }
        }
    }

    @Test
    fun leaseReturnAllowsReuseAfterScope() = runTest {
        BufferPool(capacity = 1) { TestBuffers() }.use { pool ->
            inferenceScope {
                pool.checkout()
            }
            inferenceScope {
                pool.checkout()
            }
        }
    }
}
