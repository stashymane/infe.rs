package dev.stashy.infers

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

/**
 * Maps each upstream value inside a fresh [inferenceScope].
 *
 * Prefer upstream [kotlinx.coroutines.flow.conflate] for camera / high-rate
 * sources so stale frames are dropped instead of queuing into unbounded latency.
 *
 * Native inference is not interruptible: [kotlinx.coroutines.flow.mapLatest] cannot
 * cancel an in-flight FFI call; cancellation only takes effect between frames.
 *
 * No [kotlinx.coroutines.flow.flowOn] is required — each handle dispatches onto
 * its own serialized dispatcher.
 */
public fun <T, R> Flow<T>.mapInference(transform: suspend InferenceScope.(T) -> R): Flow<R> =
    map { value -> inferenceScope { transform(value) } }
