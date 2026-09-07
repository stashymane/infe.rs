package dev.stashy.infers.xnnpack

import dev.stashy.infers.*
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.fail

class XnnpackE2eTest {
    @Test
    fun endToEndInferenceWhenModelPresent() = runTest {
        val modelPath = resolveModelPte() ?: return@runTest

        Backend().use { backend ->
            backend.loadModel(modelPath, XnnpackOptions(numThreads = 1u)).use { session ->
                assertTrue(session.inputShapes.isNotEmpty(), "expected input shapes")
                val inputShape = session.inputShapes.first()
                val dims = inputShape.dims.map { it.toInt() }
                if (dims.size != 4) {
                    fail("unexpected input rank: ${dims.size}")
                }
                val height = dims[2]
                val width = dims[3]

                val bytes = ByteArray(width * height * 3) { (it % 256).toByte() }
                val options =
                    ProcessingOptions {
                        source = width to height
                        dest = width to height
                        srcFormat = ImageFormat.Rgb888
                        destFormat = ImageFormat.Rgbf32
                    }

                CpuImageProcessor().use { processor ->
                    inferenceScope {
                        val hardware = HardwareImage.fromBytes(
                            bytes,
                            width.toUInt(),
                            height.toUInt(),
                            ImageFormat.Rgb888,
                        )
                        val pending = hardware.onCpu().process(processor, options)
                        if (pending.shape != inputShape) return@inferenceScope
                        val outputs = session.infer(pending)
                        assertTrue(outputs.isNotEmpty())
                        for (out in outputs) {
                            out.floats().use { view ->
                                assertTrue(view.size > 0)
                                for (element in view) {
                                    assertTrue(element.isFinite())
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
