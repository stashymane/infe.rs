package dev.stashy.infers

import dev.stashy.infers.internal.CloseGate
import kotlin.concurrent.atomics.AtomicBoolean
import kotlin.concurrent.atomics.ExperimentalAtomicApi

/**
 * Pool of [BufferSet] instances for overlapping [inferenceScope]s.
 *
 * Create with a fixed [capacity]; [checkout] fails immediately with
 * [InfersException.BufferPoolExhausted] when all sets are leased. Returned
 * automatically when the enclosing [InferenceScope] closes.
 */
public class BufferPool<T : BufferSet>(
    capacity: Int,
    factory: () -> T,
) : AutoCloseable {
    private val gate = CloseGate("BufferPool")
    private val lock = Any()
    private val all: List<T>
    private val free: ArrayDeque<T>

    init {
        require(capacity > 0) { "BufferPool capacity must be > 0" }
        all =
            List(capacity) {
                factory().also { it.seal() }
            }
        free = ArrayDeque(all)
    }

    public val capacity: Int
        get() = all.size

    /**
     * Lease one free buffer set for this [InferenceScope].
     *
     * Throws [InfersException.BufferPoolExhausted] if none are free.
     * The set is returned to the pool when the scope exits (or the lease is closed).
     */
    context(scope: InferenceScope)
    public fun checkout(): T {
        gate.ensureOpen()
        val set =
            synchronized(lock) {
                free.removeFirstOrNull()
                    ?: throw InfersException.BufferPoolExhausted("capacity=$capacity, all sets are in use")
            }
        scope.register(Lease(set))
        return set
    }

    private fun release(set: T) {
        set.clearImports()
        synchronized(lock) {
            if (!free.contains(set)) {
                free.addLast(set)
            }
        }
    }

    override fun close() {
        if (!gate.markClosed()) {
            return
        }
        var primary: Throwable? = null
        for (set in all) {
            try {
                set.destroySlots()
            } catch (t: Throwable) {
                if (primary == null) {
                    primary = t
                } else {
                    primary.addSuppressed(t)
                }
            }
        }
        synchronized(lock) {
            free.clear()
        }
        if (primary != null) {
            throw primary
        }
    }

    @OptIn(ExperimentalAtomicApi::class)
    private inner class Lease(
        private val set: T,
    ) : AutoCloseable {
        private val closed = AtomicBoolean(false)

        override fun close() {
            if (closed.compareAndSet(expectedValue = false, newValue = true)) {
                release(set)
            }
        }
    }
}
