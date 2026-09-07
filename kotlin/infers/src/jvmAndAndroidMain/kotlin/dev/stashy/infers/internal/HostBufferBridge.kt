package dev.stashy.infers.internal

import com.sun.jna.Pointer
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Zero-copy wrap of Rust-owned host bytes as a direct [ByteBuffer].
 *
 * The buffer does not own or free the memory; the associated [CpuTensor][dev.stashy.infers.CpuTensor]
 * (or view keep-alive) must stay open for the lifetime of the buffer.
 */
internal object HostBufferBridge {
    fun wrap(ptr: Long, nbytes: Long): ByteBuffer {
        require(nbytes >= 0) { "nbytes $nbytes < 0" }
        require(nbytes <= Int.MAX_VALUE.toLong()) { "nbytes $nbytes exceeds Int.MAX_VALUE" }
        if (nbytes == 0L) {
            return ByteBuffer.allocateDirect(0).order(ByteOrder.LITTLE_ENDIAN)
        }
        require(ptr != 0L) { "null host data pointer" }
        return Pointer(ptr)
            .getByteBuffer(0, nbytes)
            .order(ByteOrder.LITTLE_ENDIAN)
    }
}
