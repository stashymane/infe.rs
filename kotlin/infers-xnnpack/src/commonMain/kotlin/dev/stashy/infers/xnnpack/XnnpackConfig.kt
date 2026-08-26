package dev.stashy.infers.xnnpack

import dev.stashy.infers.BackendConfig
import dev.stashy.infers.internal.BackendConfigFfiConvertible
import dev.stashy.infers.ffi.BackendConfig as FfiBackendConfig

/** XNNPACK CPU backend configuration. */
public class XnnpackConfig(
    public val numThreads: UInt = 1u,
    public val method: String? = null,
) : BackendConfig,
    BackendConfigFfiConvertible {
    override fun toFfiConfig(): FfiBackendConfig =
        FfiBackendConfig.Xnnpack(
            numThreads = numThreads,
            method = method,
        )
}
