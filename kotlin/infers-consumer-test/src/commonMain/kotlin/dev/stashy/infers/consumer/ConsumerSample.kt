package dev.stashy.infers.consumer

import dev.stashy.infers.*
import kotlinx.coroutines.runBlocking

/**
 * Compilable sample that depends only on `:infers`. Succeeds only if the public
 * API is self-sufficient without a compile-time dependency on `:infers-ffi`.
 */
object ConsumerSample {
    fun run(): String =
        runBlocking {
            val device = Device.cpu()

            CpuImageProcessor().use { processor ->
                val bytes = ByteArray(48) { it.toByte() }
                val options =
                    ProcessingOptions {
                        source = 4 to 4
                        dest = 2 to 2
                        srcFormat = ImageFormat.Rgb888
                        destFormat = ImageFormat.Rgbf32
                    }
                inferenceScope {
                    tensorOf(TensorShape.of(1, 2), floatArrayOf(1f, 2f))
                    processor.process(bytes, options)
                }
            }

            Backend().use { backend ->
                backend.availableDevices()
            }

            "ok:${device.name}"
        }
}
