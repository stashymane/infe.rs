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
                val pending = hardware.onCpu().process(processor, options)
                val out = pending.materialize()
                assertEquals(TensorShape.of(1, 3, 2, 2), out.shape)
                assertEquals(DataType.F32, out.dtype)
                assertEquals(12, out.floats().use { it.size })
            }
        }
    }
}

private fun rgb4x4To2x2(): ProcessingOptions = ProcessingOptions {
    source = 4 to 4
    dest = 2 to 2
    srcFormat = ImageFormat.Rgb888
    destFormat = ImageFormat.Rgbf32
}
