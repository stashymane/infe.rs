package dev.stashy.infers

/**
 * DSL marker for [InferenceScope] so nested scopes and receivers stay unambiguous.
 */
@DslMarker
public annotation class InfersDsl

/**
 * Frame-scoped resource arena. Every [Tensor] (and on Android, [HardwareBuffer])
 * created through this scope is closed automatically when the scope exits, in
 * reverse creation order.
 *
 * Prefer reading terminal results ([Tensor.readFloats], etc.) before the scope
 * returns — a [Tensor] that escapes has already been closed.
 *
 * Native FFI calls inside the scope are not interruptible; cancellation stops
 * between stages only.
 */
@InfersDsl
public class InferenceScope internal constructor() {
    private val resources = ArrayDeque<AutoCloseable>()

    internal fun <T : AutoCloseable> register(resource: T): T {
        resources.addLast(resource)
        return resource
    }

    internal fun closeAll() {
        var primary: Throwable? = null
        while (resources.isNotEmpty()) {
            val resource = resources.removeLast()
            try {
                resource.close()
            } catch (t: Throwable) {
                if (primary == null) {
                    primary = t
                } else {
                    primary.addSuppressed(t)
                }
            }
        }
        if (primary != null) {
            throw primary
        }
    }

    /**
     * Preprocesses [bytes] with [this] processor. The returned [Tensor] is owned
     * by the scope.
     */
    public suspend fun ImageProcessor.process(bytes: ByteArray, options: ProcessingOptions): Tensor {
        val processor =
            this as? FfiBackedImageProcessor
                ?: error("ImageProcessor must be an Infers FFI-backed processor")
        return register(processor.processBytes(bytes, options))
    }

    /**
     * Runs inference with a single input. Outputs are owned by the scope.
     */
    public suspend fun ModelSession.run(input: Tensor): List<Tensor> = run(listOf(input))

    /**
     * Runs inference. Outputs are owned by the scope.
     */
    @OptIn(InfersInternalApi::class)
    public suspend fun ModelSession.run(inputs: List<Tensor>): List<Tensor> = runInternal(inputs).map { register(it) }

    /** Creates a float tensor owned by the scope. */
    public fun tensorOf(shape: TensorShape, data: FloatArray): Tensor = register(Tensor.of(shape, data))

    /** Creates a U8 tensor owned by the scope. */
    public fun tensorOf(shape: TensorShape, data: ByteArray): Tensor = register(Tensor.of(shape, data))

    /** Creates an I32 tensor owned by the scope. */
    public fun tensorOf(shape: TensorShape, data: IntArray): Tensor = register(Tensor.of(shape, data))

    /** Creates an I64 tensor owned by the scope. */
    public fun tensorOf(shape: TensorShape, data: LongArray): Tensor = register(Tensor.of(shape, data))
}

/**
 * Opens an [InferenceScope], runs [block], then closes every resource the scope
 * created — even if [block] throws.
 */
public suspend fun <R> inferenceScope(block: suspend InferenceScope.() -> R): R {
    val scope = InferenceScope()
    var blockError: Throwable? = null
    val result =
        try {
            scope.block()
        } catch (t: Throwable) {
            blockError = t
            null
        }
    try {
        scope.closeAll()
    } catch (closeError: Throwable) {
        if (blockError != null) {
            blockError.addSuppressed(closeError)
            throw blockError
        }
        throw closeError
    }
    if (blockError != null) {
        throw blockError
    }
    @Suppress("UNCHECKED_CAST")
    return result as R
}
