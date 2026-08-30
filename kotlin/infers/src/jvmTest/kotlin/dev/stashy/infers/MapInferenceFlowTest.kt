package dev.stashy.infers

import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.conflate
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.time.Duration.Companion.milliseconds

class MapInferenceFlowTest {
    @Test
    fun conflateDropsStaleFramesBeforeMapInference() = runTest {
        val collected = mutableListOf<Int>()
        val gate = Channel<Unit>(capacity = Channel.RENDEZVOUS)
        val job =
            launch {
                flow {
                    emit(1)
                    gate.receive()
                    emit(2)
                    emit(3)
                }.conflate()
                    .mapInference { it }
                    .collect { value ->
                        collected += value
                        if (value == 1) {
                            gate.send(Unit)
                            delay(1_000.milliseconds)
                        }
                    }
            }
        advanceUntilIdle()
        job.cancel()
        // Collector takes 1, then 2 and 3 arrive while it is delayed; conflate keeps 3.
        assertEquals(listOf(1, 3), collected)
    }
}
