package dev.stashy.infers.internal

import dev.stashy.infers.InfersInternalApi
import kotlin.concurrent.atomics.AtomicBoolean
import kotlin.concurrent.atomics.ExperimentalAtomicApi

/** Idempotent close gate shared by public AutoCloseable wrappers. */
@InfersInternalApi
@OptIn(ExperimentalAtomicApi::class)
public class CloseGate(
    private val owner: String,
) {
    private val closed = AtomicBoolean(false)

    public fun ensureOpen() {
        check(!closed.load()) { "$owner is closed" }
    }

    /** Returns true the first time close wins; subsequent calls are no-ops. */
    public fun markClosed(): Boolean = closed.compareAndSet(expectedValue = false, newValue = true)
}
