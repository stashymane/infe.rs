package dev.stashy.infers

import dev.stashy.infers.ffi.FfiBackend
import dev.stashy.infers.internal.CloseGate

/** Entry point for loading ExecuTorch models. Feature modules provide typed loaders. */
public class Backend : AutoCloseable {
    @InfersInternalApi
    public val handle: FfiBackend = FfiBackend()
    private val gate = CloseGate("Backend")

    @InfersInternalApi
    public fun ensureOpen() {
        gate.ensureOpen()
    }

    override fun close() {
        if (gate.markClosed()) {
            handle.close()
        }
    }
}
