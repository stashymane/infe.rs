package dev.stashy.infers

internal actual class PlatformFrameImport actual constructor() {
    actual fun hasImport(): Boolean = false

    actual fun clear() {}

    actual fun copyIntoHostIfNeeded(host: HardwareImage) {}
}
