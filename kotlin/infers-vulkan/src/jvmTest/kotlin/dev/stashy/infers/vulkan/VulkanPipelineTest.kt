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
        val device = gpuDeviceOrNull() ?: return@runTest
        device.use {
            GpuImageProcessor(it).use { processor ->
                assertEquals(DeviceKind.Gpu, processor.deviceInfo.kind)
            }
        }
    }

    @Test
    fun gpuMatchesCpuOutputShape() = runTest {
        val device = gpuDeviceOrNull() ?: return@runTest
        val rawBytes = ByteArray(48) { (it * 5).toByte() }
        val options =
            ProcessingOptions {
                source = 4 to 4
                dest = 2 to 2
                srcFormat = ImageFormat.Rgb888
                destFormat = ImageFormat.Rgbf32
            }

        device.use {
            CpuImageProcessor().use { cpu ->
                GpuImageProcessor(it).use { gpu ->
                    val cpuShape =
                        inferenceScope {
                            val hardware = HardwareImage.fromBytes(rawBytes, 4u, 4u, ImageFormat.Rgb888)
                            hardware.on(CpuDevice).process(cpu, options).materialize().shape
                        }
                    inferenceScope {
                        val hardware = HardwareImage.fromBytes(rawBytes, 4u, 4u, ImageFormat.Rgb888)
                        val pending = hardware.on(it).process(gpu, options)
                        val out = pending.materialize()
                        assertEquals(cpuShape, out.shape)
                        assertEquals(12, out.floats().use { it.size })
                    }
                }
            }
        }
    }

    @Test
    fun invalidModelThrowsModelLoadFailed() = runTest {
        val device = gpuDeviceOrNull() ?: return@runTest
        device.use {
            Backend().use { backend ->
                val path = Path(SystemTemporaryDirectory, "infers-invalid-vulkan.pte")
                SystemFileSystem.sink(path).buffered().use { sink ->
                    sink.write(byteArrayOf(0xDE.toByte(), 0xAD.toByte()))
                }
                try {
                    val error = assertFailsWith<InfersException.ModelLoadFailed> {
                        backend.loadModel(path, it, VulkanOptions())
                    }
                    assertTrue(error.reason.isNotEmpty())
                } finally {
                    runCatching { SystemFileSystem.delete(path, mustExist = false) }
                }
            }
        }
    }
}

private fun gpuDeviceOrNull(): GpuDevice? = runCatching { GpuDevice(0u) }.getOrNull()
