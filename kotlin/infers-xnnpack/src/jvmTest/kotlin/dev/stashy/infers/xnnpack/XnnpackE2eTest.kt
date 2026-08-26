package dev.stashy.infers.xnnpack

import dev.stashy.infers.Backend
import dev.stashy.infers.CpuImageProcessor
import dev.stashy.infers.ImageFormat
import dev.stashy.infers.ProcessingOptions
import dev.stashy.infers.inferenceScope
import kotlinx.coroutines.test.runTest
import kotlinx.io.files.Path
import org.junit.Assume.assumeTrue
import java.io.File
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.fail

class XnnpackE2eTest {
    @Test
    fun endToEndInferenceWhenModelPresent() =
        runTest {
            val modelFile = resolveModelPte()
            assumeTrue("model.pte not present; generate assets first", modelFile != null)

            Backend().use { backend ->
                backend.loadModel(Path(modelFile!!.absolutePath), XnnpackConfig(numThreads = 1u)).use { session ->
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
                            val input = processor.process(bytes, options)
                            assumeTrue(
                                "preprocessed shape ${input.shape} != model input $inputShape",
                                input.shape == inputShape,
                            )
                            val outputs = session.run(listOf(input))
                            assertTrue(outputs.isNotEmpty())
                            for (out in outputs) {
                                val floats = out.readFloats()
                                assertTrue(floats.isNotEmpty())
                                assertTrue(floats.all { v -> v.isFinite() })
                            }
                        }
                    }
                }
            }
        }

    private fun resolveModelPte(): File? {
        val injected = System.getProperty("infers.model.xnnpack")
        if (injected != null) {
            val file = File(injected)
            if (file.isFile) return file
        }
        val candidates =
            listOf(
                File("../../assets/yolo26n-face/xnnpack/model.pte"),
                File("../assets/yolo26n-face/xnnpack/model.pte"),
                File("assets/yolo26n-face/xnnpack/model.pte"),
                File(System.getProperty("user.dir"), "../assets/yolo26n-face/xnnpack/model.pte"),
            )
        return candidates.firstOrNull { it.isFile }
    }
}
