package dev.stashy.infers.sample

import dev.stashy.infers.*
import dev.stashy.infers.vulkan.*
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.runBlocking
import kotlinx.io.files.Path

/** Synthetic camera frame (RGB888 bytes already resident, as from a device buffer). */
internal data class CameraFrame(
    val index: Long,
    val width: UInt,
    val height: UInt,
    val bytes: ByteArray,
)

/** Inference output for one frame: tensor metadata plus a short float preview. */
internal data class GpuPipelineOutput(
    val frameIndex: Long,
    val outputTensorCount: Int,
    val firstOutputShape: TensorShape?,
    val firstOutputPreview: FloatArray,
)

private class PipelineBuffers(
    device: GpuDevice,
    width: UInt,
    height: UInt,
) : BufferSet() {
    val frame by buffer(device, width, height, ImageFormat.Rgb888)
}

/**
 * GPU preprocess → Vulkan ExecuTorch inference, driven by a [Flow] of [CameraFrame]s.
 *
 * Long-lived handles ([GpuDevice], session, processor, [BufferPool]) stay open for
 * the pipeline lifetime. Per-frame tensors are owned by a fresh
 * [dev.stashy.infers.inferenceScope] inside [mapInference]; frame bytes reuse a
 * pooled [FrameBuffer].
 */
internal class GpuPipeline private constructor(
    private val device: GpuDevice,
    private val backend: Backend,
    private val session: GpuSession,
    private val processor: GpuImageProcessor,
    private val processingOptions: ProcessingOptions,
    private val buffers: BufferPool<PipelineBuffers>,
) : AutoCloseable {
    fun processFrames(frames: Flow<CameraFrame>): Flow<GpuPipelineOutput> = frames.mapInference { frame ->
        val slotted = buffers.checkout()
        slotted.frame.write(frame.bytes)
        val pending = slotted.frame.on().process(processor, processingOptions)
        val outputs = session.infer(pending)
        val primary = outputs.firstOrNull()
        val preview = primary?.floats()?.use { view ->
            val n = minOf(PREVIEW_FLOAT_COUNT, view.size)
            FloatArray(n).also { view.copyInto(it, endIndex = n) }
        } ?: floatArrayOf()

        GpuPipelineOutput(
            frameIndex = frame.index,
            outputTensorCount = outputs.size,
            firstOutputShape = primary?.shape,
            firstOutputPreview = preview,
        )
    }

    override fun close() {
        buffers.close()
        processor.close()
        session.close()
        backend.close()
        device.close()
    }

    companion object {
        private const val PREVIEW_FLOAT_COUNT = 8
        private const val BUFFER_POOL_CAPACITY = 2

        fun open(modelPath: Path, frameWidth: UInt, frameHeight: UInt): GpuPipeline? {
            val device = runCatching { GpuDevice(0u) }.getOrNull() ?: return null
            val backend = Backend()
            return try {
                val session = runBlocking {
                    backend.loadModel(modelPath, device, VulkanOptions())
                }
                val processor = GpuImageProcessor(device)
                val options = ProcessingOptions.forModelInput(session) {
                    source = frameWidth.toInt() to frameHeight.toInt()
                    srcFormat = ImageFormat.Rgb888
                }
                val buffers =
                    BufferPool(BUFFER_POOL_CAPACITY) {
                        PipelineBuffers(device, frameWidth, frameHeight)
                    }
                GpuPipeline(device, backend, session, processor, options, buffers)
            } catch (_: Throwable) {
                backend.close()
                device.close()
                null
            }
        }
    }
}
