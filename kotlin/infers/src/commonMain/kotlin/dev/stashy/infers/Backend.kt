package dev.stashy.infers

import dev.stashy.infers.ffi.FfiBackend
import dev.stashy.infers.internal.CloseGate
import dev.stashy.infers.internal.fromFfi
import dev.stashy.infers.internal.toFfi
import dev.stashy.infers.internal.withFfiErrors
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.io.files.Path

public class Backend : AutoCloseable {
    private val handle = FfiBackend()
    private val gate = CloseGate("Backend")

    /**
     * Lists devices reported by the native backend.
     *
     * Native calls are not interruptible; cancellation is checked around the call.
     */
    public suspend fun availableDevices(): List<Device> =
        withContext(Dispatchers.Default) {
            gate.ensureOpen()
            handle.availableDevices().map { it.fromFfi() }
        }

    /**
     * Loads a `.pte` from [path]. The file must remain readable for the lifetime
     * of the returned [ModelSession]. Callers that only have a stream or byte
     * buffer should write their own file before calling this.
     *
     * Runs on [Dispatchers.IO]. Native load is not interruptible mid-call.
     */
    public suspend fun loadModel(
        path: Path,
        config: BackendConfig,
    ): ModelSession =
        withContext(Dispatchers.IO) {
            withFfiErrors {
                gate.ensureOpen()
                ModelSession(handle.loadModelFromFile(path.toString(), config.toFfi()))
            }
        }

    override fun close() {
        if (gate.markClosed()) {
            handle.close()
        }
    }
}
