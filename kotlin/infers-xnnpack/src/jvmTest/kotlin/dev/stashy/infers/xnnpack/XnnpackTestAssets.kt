package dev.stashy.infers.xnnpack

import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem

internal fun resolveModelPte(): Path? {
    System.getProperty("infers.model.xnnpack")?.let { injected ->
        val path = Path(injected)
        if (SystemFileSystem.exists(path)) return path
    }
    val userDir = System.getProperty("user.dir") ?: return null
    val candidates =
        listOf(
            Path("../../target/yolo26n-face/xnnpack/model.pte"),
            Path("../target/yolo26n-face/xnnpack/model.pte"),
            Path("target/yolo26n-face/xnnpack/model.pte"),
            Path(userDir, "../target/yolo26n-face/xnnpack/model.pte"),
        )
    return candidates.firstOrNull { SystemFileSystem.exists(it) }
}
