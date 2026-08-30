package dev.stashy.infers.xnnpack

import dev.stashy.infers.Backend
import dev.stashy.infers.InfersException
import kotlinx.coroutines.test.runTest
import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import kotlinx.io.files.SystemTemporaryDirectory
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class XnnpackBackendTest {
    @Test
    fun availableDevicesIsNonEmpty() = runTest {
        Backend().use { backend ->
            assertTrue(backend.availableDevices().isNotEmpty())
        }
    }

    @Test
    fun invalidModelThrowsModelLoadFailed() = runTest {
        Backend().use { backend ->
            val path = Path(SystemTemporaryDirectory, "infers-invalid-xnnpack.pte")
            SystemFileSystem.sink(path).buffered().use { sink ->
                sink.write(byteArrayOf(0xDE.toByte(), 0xAD.toByte()))
            }
            try {
                val error =
                    assertFailsWith<InfersException.ModelLoadFailed> {
                        backend.loadModel(path, XnnpackConfig(numThreads = 1u))
                    }
                assertTrue(error.reason.isNotEmpty())
            } finally {
                runCatching { SystemFileSystem.delete(path, mustExist = false) }
            }
        }
    }
}
