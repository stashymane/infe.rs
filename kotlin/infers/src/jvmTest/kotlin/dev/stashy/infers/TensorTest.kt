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
            assertTrue(tensor.readFloats().contentEquals(data))
        }
    }

    @Test
    fun closedTensorThrows() = runTest {
        val tensor = inferenceScope {
            cpuTensorOf(TensorShape.of(1), floatArrayOf(1f))
        }
        assertFailsWith<IllegalStateException> {
            tensor.readFloats()
        }
    }
}
