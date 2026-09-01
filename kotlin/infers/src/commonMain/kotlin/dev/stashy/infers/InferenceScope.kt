package dev.stashy.infers

/**
 * DSL marker for [InferenceScope] so nested scopes and receivers stay unambiguous.
 */
@DslMarker
public annotation class InfersDsl

/**
 * Frame-scoped resource arena. Every [Tensor], [HostImage], and on Android
 * [HardwareBuffer] created through this scope is closed automatically when the
 * scope exits, in reverse creation order.
 *
 * Preprocess, upload, and inference ([infer], [ImageProcessor.process]) require this
 * scope via context parameters and cannot be called outside [inferenceScope].
 */
@InfersDsl
public class InferenceScope internal constructor() {
    private val resources = ArrayDeque<AutoCloseable>()

    @InfersInternalApi
    public fun <T : AutoCloseable> register(resource: T): T {
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

    /** Creates a float CPU tensor owned by the scope. */
    public fun cpuTensorOf(shape: TensorShape, data: FloatArray): CpuTensor = register(CpuTensor.of(shape, data))

    /** Creates a U8 CPU tensor owned by the scope. */
    public fun cpuTensorOf(shape: TensorShape, data: ByteArray): CpuTensor = register(CpuTensor.of(shape, data))

    /** Creates an I32 CPU tensor owned by the scope. */
    public fun cpuTensorOf(shape: TensorShape, data: IntArray): CpuTensor = register(CpuTensor.of(shape, data))

    /** Creates an I64 CPU tensor owned by the scope. */
    public fun cpuTensorOf(shape: TensorShape, data: LongArray): CpuTensor = register(CpuTensor.of(shape, data))
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
