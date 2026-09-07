package dev.stashy.infers

import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class TensorTest {
    @Test
    fun floatTensorRoundTrip() = runTest {
        inferenceScope {
            val shape = TensorShape.of(1, 3, 2, 2)
            val data = FloatArray(12) { it.toFloat() }
            val tensor = cpuTensorOf(shape, data)
            assertEquals(shape, tensor.shape)
            assertEquals(DataType.F32, tensor.dtype)
            assertEquals(48u, tensor.byteSize)
            tensor.floats().use { view ->
                val out = FloatArray(view.size)
                view.copyInto(out)
                assertTrue(out.contentEquals(data))
            }
        }
    }

    @Test
    fun floatViewRegionCopy() = runTest {
        inferenceScope {
            val data = FloatArray(12) { it.toFloat() }
            val tensor = cpuTensorOf(TensorShape.of(1, 3, 2, 2), data)
            tensor.floats().use { view ->
                assertEquals(12, view.size)
                assertEquals(3f, view[3])
                val region = FloatArray(4)
                view.copyInto(region, startIndex = 4, endIndex = 8)
                assertTrue(region.contentEquals(floatArrayOf(4f, 5f, 6f, 7f)))
            }
        }
    }

    @Test
    fun floatViewAsBufferZeroCopy() = runTest {
        inferenceScope {
            val data = FloatArray(4) { it.toFloat() * 2f }
            val tensor = cpuTensorOf(TensorShape.of(4), data)
            tensor.floats().use { view ->
                val buf = view.asBuffer()
                assertEquals(4, buf.remaining())
                assertEquals(0f, buf.get(0))
                assertEquals(6f, buf.get(3))
            }
        }
    }

    @Test
    fun floatViewIsIterable() = runTest {
        inferenceScope {
            val data = floatArrayOf(1f, 2f, 3f, 4f)
            val tensor = cpuTensorOf(TensorShape.of(4), data)
            tensor.floats().use { view ->
                val scanned = ArrayList<Float>(view.size)
                for (v in view) {
                    scanned.add(v)
                }
                assertEquals(data.toList(), scanned)
            }
        }
    }

    @Test
    fun closedTensorThrows() = runTest {
        val tensor = inferenceScope {
            cpuTensorOf(TensorShape.of(1), floatArrayOf(1f))
        }
        assertFailsWith<IllegalStateException> {
            tensor.floats()
        }
    }
}
