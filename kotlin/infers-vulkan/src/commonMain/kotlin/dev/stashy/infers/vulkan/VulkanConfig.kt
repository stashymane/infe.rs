package dev.stashy.infers.vulkan

import dev.stashy.infers.BackendConfig
import dev.stashy.infers.internal.BackendConfigFfiConvertible
import dev.stashy.infers.ffi.BackendConfig as FfiBackendConfig

/** ExecuTorch Vulkan backend configuration, sharing [context] with GPU processors. */
public class VulkanConfig(
    public val context: GpuContext,
    public val method: String? = null,
) : BackendConfig,
    BackendConfigFfiConvertible {
    override fun toFfiConfig(): FfiBackendConfig = FfiBackendConfig.Vulkan(
        context = context.handle,
        method = method,
    )
}
