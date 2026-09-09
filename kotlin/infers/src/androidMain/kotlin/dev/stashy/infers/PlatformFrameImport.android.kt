package dev.stashy.infers

import java.util.concurrent.atomic.AtomicReference

internal actual class PlatformFrameImport actual constructor() {
    private val imported = AtomicReference<HardwareBuffer?>(null)

    actual fun hasImport(): Boolean = imported.get() != null

    actual fun clear() {
        imported.getAndSet(null)?.close()
    }

    actual fun copyIntoHostIfNeeded(host: HardwareImage) {
        val buffer = imported.get() ?: return
        host.write(buffer.lockCpu())
    }

    fun install(buffer: HardwareBuffer) {
        imported.getAndSet(buffer)?.close()
    }

    fun peek(): HardwareBuffer? = imported.get()
}
