package dev.stashy.infers

import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals

class CpuImageProcessorTest {
    @Test
    fun resizeRgb888ToRgbf32() = runTest {
        val rawBytes = ByteArray(48) { (it * 5).toByte() }
        CpuImageProcessor().use { processor ->
            assertEquals(DeviceKind.Cpu, processor.device.kind)
            inferenceScope {
                val out = processor.process(rawBytes, rgb4x4To2x2())
                assertEquals(TensorShape.of(1, 2, 2, 3), out.shape)
                assertEquals(DataType.F32, out.dtype)
                assertEquals(12, out.readFloats().size)
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
