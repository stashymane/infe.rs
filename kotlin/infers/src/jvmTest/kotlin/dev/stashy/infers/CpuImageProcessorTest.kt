package dev.stashy.infers

import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals

class CpuImageProcessorTest {
    @Test
    fun resizeRgb888ToRgbf32() = runTest {
        val rawBytes = ByteArray(48) { (it * 5).toByte() }
        CpuImageProcessor().use { processor ->
            assertEquals(DeviceKind.Cpu, processor.deviceInfo.kind)
            inferenceScope {
                val options = rgb4x4To2x2()
                val hardware = HardwareImage.fromBytes(rawBytes, 4u, 4u, ImageFormat.Rgb888)
                val pending = hardware.on(CpuDevice).process(processor, options)
                val out = pending.materialize()
                assertEquals(TensorShape.of(1, 3, 2, 2), out.shape)
                assertEquals(DataType.F32, out.dtype)
                assertEquals(12, out.floats().use { it.size })
            }
        }
    }

    @Test
    fun deferredMaterializesCpuImage() = runTest {
        val rawBytes = ByteArray(48) { (it * 5).toByte() }
        inferenceScope {
            val hardware = HardwareImage.fromBytes(rawBytes, 4u, 4u, ImageFormat.Rgb888)
            val image: DeviceImage<CpuDevice> = hardware.on(CpuDevice).materialize()
            assertEquals(4u, image.width)
            assertEquals(4u, image.height)
            assertEquals(ImageFormat.Rgb888, image.format)
        }
    }
}

private fun rgb4x4To2x2(): ProcessingOptions = ProcessingOptions {
    source = 4 to 4
    dest = 2 to 2
    srcFormat = ImageFormat.Rgb888
    destFormat = ImageFormat.Rgbf32
}
