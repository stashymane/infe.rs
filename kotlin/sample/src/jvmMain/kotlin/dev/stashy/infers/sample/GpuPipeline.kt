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

/**
 * GPU preprocess → Vulkan ExecuTorch inference, driven by a [Flow] of [CameraFrame]s.
 *
 * Long-lived handles ([GpuDevice], session, processor) stay open for the pipeline lifetime.
 * Per-frame tensors and images are owned by a fresh [dev.stashy.infers.inferenceScope] inside
 * [mapInference].
 */
internal class GpuPipeline private constructor(
    private val device: GpuDevice,
    private val backend: Backend,
    private val session: GpuSession,
    private val processor: GpuImageProcessor,
    private val processingOptions: ProcessingOptions,
) : AutoCloseable {
    fun processFrames(frames: Flow<CameraFrame>): Flow<GpuPipelineOutput> = frames.mapInference { frame ->
        val hardware = HardwareImage.fromBytes(frame.bytes, frame.width, frame.height, ImageFormat.Rgb888)
        val pending = hardware.on(device).process(processor, processingOptions)
        val outputs = session.infer(pending)
        val primary = outputs.firstOrNull()
        val preview = primary
            ?.readFloats()
            ?.take(PREVIEW_FLOAT_COUNT)
            .orEmpty()
            .toFloatArray()

        GpuPipelineOutput(
            frameIndex = frame.index,
            outputTensorCount = outputs.size,
            firstOutputShape = primary?.shape,
            firstOutputPreview = preview,
        )
    }

    override fun close() {
        processor.close()
        session.close()
        backend.close()
        device.close()
    }

    companion object {
        private const val PREVIEW_FLOAT_COUNT = 8

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
                GpuPipeline(device, backend, session, processor, options)
            } catch (_: Throwable) {
                backend.close()
                device.close()
                null
            }
        }

        private fun imgszFromInputShape(shape: TensorShape): Int {
            val dims = shape.dims.map { it.toInt() }
            require(dims.size == 4 && dims[0] == 1 && dims[1] == 3 && dims[2] == dims[3]) {
                "expected square NCHW input [1, 3, H, H], got $dims"
            }
            return dims[2]
        }
    }
}
