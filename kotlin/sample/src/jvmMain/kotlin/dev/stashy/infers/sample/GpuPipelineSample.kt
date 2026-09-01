package dev.stashy.infers.sample

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.runBlocking

/**
 * Demonstrates a GPU pipeline: synthetic camera frames → GPU preprocess → Vulkan inference →
 * [Flow] of output summaries.
 *
 * Requires a Vulkan GPU and `target/yolo26n-face/vulkan/model.pte` (see `scripts/build_yolo26n_face.sh`).
 * Override the model path with `-Dinfers.model.vulkan=/path/to/model.pte`.
 */
fun main() {
    val modelPath = resolveVulkanModelPte() ?: error(
        "Vulkan model not found. Build assets (scripts/build_yolo26n_face.sh) or set " +
            "-Dinfers.model.vulkan=/path/to/model.pte",
    )

    val frameWidth = 1280u
    val frameHeight = 720u
    val frameCount = 3

    val pipeline = GpuPipeline.open(modelPath, frameWidth, frameHeight) ?: error(
        "Failed to open GPU pipeline (Vulkan unavailable or model load failed).",
    )

    pipeline.use {
        runBlocking {
            it.processFrames(syntheticCameraFrames(frameWidth, frameHeight, frameCount)).collect { output ->
                val shape = output.firstOutputShape?.dims?.joinToString(prefix = "[", postfix = "]") ?: "none"
                val preview = output.firstOutputPreview.joinToString(prefix = "[", postfix = "]")
                println(
                    "frame ${output.frameIndex}: outputs=${output.outputTensorCount} " +
                        "first_shape=$shape preview=$preview",
                )
            }
        }
    }
}

internal fun syntheticCameraFrames(width: UInt, height: UInt, count: Int): Flow<CameraFrame> = flow {
    for (index in 0 until count) {
        emit(syntheticCameraFrame(index.toLong(), width, height))
    }
}

internal fun syntheticCameraFrame(index: Long, width: UInt, height: UInt): CameraFrame {
    val w = width.toInt()
    val h = height.toInt()
    val bytes = ByteArray(w * h * 3)
    var offset = 0
    for (y in 0 until h) {
        for (x in 0 until w) {
            bytes[offset++] = ((x * 255) / w.coerceAtLeast(1)).toByte()
            bytes[offset++] = ((y * 255) / h.coerceAtLeast(1)).toByte()
            bytes[offset++] = (((x + y) * 255) / (w + h).coerceAtLeast(1)).toByte()
        }
    }
    return CameraFrame(index = index, width = width, height = height, bytes = bytes)
}
