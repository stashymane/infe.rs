package dev.stashy.infers.vulkan

import dev.stashy.infers.*
import kotlinx.coroutines.test.runTest
import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import kotlinx.io.files.SystemTemporaryDirectory
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class VulkanPipelineTest {
    @Test
    fun gpuImageProcessorDeviceKind() = runTest {
        val context = gpuContextOrNull() ?: return@runTest
        context.use {
            GpuImageProcessor(it).use { processor ->
                assertEquals(DeviceKind.Gpu, processor.device.kind)
            }
        }
    }

    @Test
    fun gpuMatchesCpuOutputShape() = runTest {
        val context = gpuContextOrNull() ?: return@runTest
        val rawBytes = ByteArray(48) { (it * 5).toByte() }
        val options =
            ProcessingOptions {
                source = 4 to 4
                dest = 2 to 2
                srcFormat = ImageFormat.Rgb888
                destFormat = ImageFormat.Rgbf32
            }

        context.use {
            CpuImageProcessor().use { cpu ->
                GpuImageProcessor(it).use { gpu ->
                    val cpuShape =
                        inferenceScope {
                            cpu.process(rawBytes, options).shape
                        }
                    inferenceScope {
                        val out = gpu.process(rawBytes, options)
                        assertEquals(cpuShape, out.shape)
                        assertEquals(12, out.readFloats().size)
                    }
                }
            }
        }
    }

    @Test
    fun invalidModelThrowsModelLoadFailed() = runTest {
        val context = gpuContextOrNull() ?: return@runTest
        context.use {
            Backend().use { backend ->
                val path = Path(SystemTemporaryDirectory, "infers-invalid-vulkan.pte")
                SystemFileSystem.sink(path).buffered().use { sink ->
                    sink.write(byteArrayOf(0xDE.toByte(), 0xAD.toByte()))
                }
                try {
                    val error = assertFailsWith<InfersException.ModelLoadFailed> {
                        backend.loadModel(path, VulkanConfig(it))
                    }
                    assertTrue(error.reason.isNotEmpty())
                } finally {
                    runCatching { SystemFileSystem.delete(path, mustExist = false) }
                }
            }
        }
    }
}

private fun gpuContextOrNull(): GpuContext? = runCatching { GpuContext(Device.gpu(0u)) }.getOrNull()
