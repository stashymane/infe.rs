package dev.stashy.infers.sample

import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem

internal fun resolveVulkanModelPte(): Path? {
    System.getProperty("infers.model.vulkan")?.let { injected ->
        val path = Path(injected)
        if (SystemFileSystem.exists(path)) return path
    }
    val userDir = System.getProperty("user.dir") ?: return null
    val candidates =
        listOf(
            Path("../../target/yolo26n-face/vulkan/model.pte"),
            Path("../target/yolo26n-face/vulkan/model.pte"),
            Path("target/yolo26n-face/vulkan/model.pte"),
            Path(userDir, "../target/yolo26n-face/vulkan/model.pte"),
        )
    return candidates.firstOrNull { SystemFileSystem.exists(it) }
}
