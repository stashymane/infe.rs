package dev.stashy.infers

import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals

class MapInferenceTest {
    @Test
    fun mapInferenceRunsInsideScope() = runTest {
        val results =
            flowOf(1, 2, 3)
                .mapInference { n ->
                    cpuTensorOf(TensorShape.of(1), floatArrayOf(n.toFloat())).floats().use { it[0] }
                }.toList()
        assertEquals(listOf(1f, 2f, 3f), results)
    }
}
