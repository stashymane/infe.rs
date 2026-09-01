package dev.stashy.infers

import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertFailsWith

class InferenceScopeTest {
    @Test
    fun closesResourcesWhenBlockThrows() = runTest {
        assertFailsWith<IllegalStateException> {
            inferenceScope {
                cpuTensorOf(TensorShape.of(1), floatArrayOf(1f))
                error("boom")
            }
        }
        val escaped =
            inferenceScope {
                cpuTensorOf(TensorShape.of(1), floatArrayOf(2f))
            }
        assertFailsWith<IllegalStateException> {
            escaped.readBytes()
        }
    }

    @Test
    fun closesEvenWhenBlockSucceeds() = runTest {
        val tensor =
            inferenceScope {
                cpuTensorOf(TensorShape.of(2), floatArrayOf(1f, 2f))
            }
        assertFailsWith<IllegalStateException> {
            tensor.readFloats()
        }
    }
}
