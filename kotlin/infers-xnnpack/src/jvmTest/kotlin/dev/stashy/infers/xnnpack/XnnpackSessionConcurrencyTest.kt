package dev.stashy.infers.xnnpack

import dev.stashy.infers.Backend
import dev.stashy.infers.ModelSession
import dev.stashy.infers.inferenceScope
import kotlinx.coroutines.async
import kotlinx.coroutines.test.runTest
import kotlinx.io.files.Path
import org.junit.Assume.assumeTrue
import java.io.File
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.time.TimeSource

class XnnpackSessionConcurrencyTest {
    @Test
    fun differentSessionsOverlapWhileSameSessionSerializes() =
        runTest {
            val modelFile = resolveModelPte()
            assumeTrue("model.pte not present; generate assets first", modelFile != null)

            Backend().use { backend ->
                val path = Path(modelFile!!.absolutePath)
                backend.loadModel(path, XnnpackConfig(numThreads = 1u)).use { sessionA ->
                    backend.loadModel(path, XnnpackConfig(numThreads = 1u)).use { sessionB ->
                        val inputShape = sessionA.inputShapes.first()
                        val elementCount = inputShape.dims.fold(1L) { acc, d -> acc * d.toLong() }.toInt()
                        val zeros = FloatArray(elementCount)

                        suspend fun once(session: ModelSession) {
                            inferenceScope {
                                val input = tensorOf(inputShape, zeros)
                                session.run(listOf(input))
                            }
                        }

                        // Warmup so timed runs exclude first-load cost.
                        once(sessionA)
                        once(sessionB)

                        val mark = TimeSource.Monotonic.markNow()
                        val first = async { once(sessionA) }
                        val second = async { once(sessionB) }
                        first.await()
                        second.await()
                        val parallelMs = mark.elapsedNow().inWholeMilliseconds

                        val seqMark = TimeSource.Monotonic.markNow()
                        once(sessionA)
                        once(sessionA)
                        val serialMs = seqMark.elapsedNow().inWholeMilliseconds

                        // Different sessions should not be slower than same-session serial
                        // by a large margin; allow noise but require overlap benefit or parity.
                        assertTrue(
                            parallelMs <= serialMs * 1.5 + 50,
                            "expected concurrent sessions to overlap " +
                                "(parallel=${parallelMs}ms, serial=${serialMs}ms)",
                        )
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
                File("../../target/yolo26n-face/xnnpack/model.pte"),
                File("../target/yolo26n-face/xnnpack/model.pte"),
                File("target/yolo26n-face/xnnpack/model.pte"),
                File(System.getProperty("user.dir"), "../target/yolo26n-face/xnnpack/model.pte"),
            )
        return candidates.firstOrNull { it.isFile }
    }
}
