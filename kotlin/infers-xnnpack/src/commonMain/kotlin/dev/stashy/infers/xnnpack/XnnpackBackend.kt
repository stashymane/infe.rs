package dev.stashy.infers.xnnpack

import dev.stashy.infers.Backend
import dev.stashy.infers.CpuSession
import dev.stashy.infers.internal.withFfiErrors
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.io.files.Path
import dev.stashy.infers.ffi.XnnpackOptions as FfiXnnpackOptions

/** XNNPACK CPU backend options. */
public class XnnpackOptions(
    public val numThreads: UInt = 1u,
    public val method: String? = null,
) {
    internal fun toFfi(): FfiXnnpackOptions = FfiXnnpackOptions(
        numThreads = numThreads,
        method = method,
    )
}

/** Loads a `.pte` model for CPU (XNNPACK) inference. */
public suspend fun Backend.loadModel(path: Path, options: XnnpackOptions = XnnpackOptions()): CpuSession =
    withContext(Dispatchers.IO) {
        withFfiErrors {
            ensureOpen()
            CpuSession(handle.loadXnnpackFromFile(path.toString(), options.toFfi()))
        }
    }
