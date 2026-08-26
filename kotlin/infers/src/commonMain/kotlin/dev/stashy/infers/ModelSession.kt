package dev.stashy.infers

import dev.stashy.infers.internal.CloseGate
import dev.stashy.infers.internal.fromFfi
import dev.stashy.infers.internal.withFfiErrors
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.withContext
import dev.stashy.infers.ffi.ModelSession as FfiModelSession

/**
 * Loaded model ready for inference. Own this for the lifetime of a feature;
 * use [inferenceScope] for per-frame tensors.
 *
 * Each session owns a single-threaded dispatcher matching the native mutex.
 * Concurrent [runInternal] calls on the same session serialize; different
 * sessions run in parallel.
 *
 * Native calls are not interruptible — cancellation is observed between stages.
 */
@OptIn(ExperimentalCoroutinesApi::class)
public class ModelSession internal constructor(
    private val handle: FfiModelSession,
    private val dispatcher: CoroutineDispatcher = Dispatchers.Default.limitedParallelism(1),
) : AutoCloseable {
    private val gate = CloseGate("ModelSession")

    public val device: Device
        get() {
            gate.ensureOpen()
            return handle.device().fromFfi()
        }

    public val inputShapes: List<TensorShape>
        get() {
            gate.ensureOpen()
            return handle.inputShapes().map { it.fromFfi() }
        }

    public val outputShapes: List<TensorShape>
        get() {
            gate.ensureOpen()
            return handle.outputShapes().map { it.fromFfi() }
        }

    /**
     * Runs inference. Prefer calling via [InferenceScope.run] so outputs are
     * registered for automatic cleanup.
     *
     * Native calls are not interruptible; cancellation is checked around the call.
     */
    @InfersInternalApi
    public suspend fun runInternal(inputs: List<Tensor>): List<Tensor> =
        withContext(dispatcher) {
            withFfiErrors {
                gate.ensureOpen()
                handle.run(inputs.map { it.handle }).map { Tensor.fromFfi(it) }
            }
        }

    override fun close() {
        if (gate.markClosed()) {
            handle.close()
        }
    }
}
