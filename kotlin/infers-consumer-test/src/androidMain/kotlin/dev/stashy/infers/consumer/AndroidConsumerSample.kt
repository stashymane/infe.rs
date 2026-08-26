package dev.stashy.infers.consumer

import dev.stashy.infers.Device
import dev.stashy.infers.HardwareBuffer

/**
 * Compiles only for Android — proves HardwareBuffer arrives with `:infers`
 * without a separate platform module.
 */
object AndroidConsumerSample {
    fun describe(): String {
        // Reference the type so the compiler must resolve it.
        val name = HardwareBuffer::class.simpleName
        val device = Device.cpu()
        return "$name on ${device.name}"
    }
}
