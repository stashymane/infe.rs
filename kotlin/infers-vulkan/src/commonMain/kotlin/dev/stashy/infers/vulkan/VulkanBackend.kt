package dev.stashy.infers.vulkan

import dev.stashy.infers.Backend
import dev.stashy.infers.internal.toFfi
import dev.stashy.infers.internal.withFfiErrors
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.io.files.Path
import dev.stashy.infers.ffi.VulkanOptions as FfiVulkanOptions

/** Vulkan delegate options for ExecuTorch GPU inference. */
public class VulkanOptions(
    public val method: String? = null,
) {
    internal fun toFfi(): FfiVulkanOptions = FfiVulkanOptions(method = method)
}

public suspend fun Backend.loadModel(
    path: Path,
    device: GpuDevice,
    options: VulkanOptions = VulkanOptions(),
): GpuSession = withContext(Dispatchers.IO) {
    withFfiErrors {
        ensureOpen()
        GpuSession(handle.loadVulkanFromFile(path.toString(), device.handle, options.toFfi()))
    }
}
