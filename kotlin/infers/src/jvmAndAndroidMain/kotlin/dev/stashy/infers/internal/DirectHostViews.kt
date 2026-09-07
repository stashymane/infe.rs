package dev.stashy.infers.internal

import dev.stashy.infers.*
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer
import dev.stashy.infers.ffi.CpuTensor as FfiCpuTensor

/** DirectByteBuffer-backed host views (zero-copy after ET→Arc materialization). */
internal object DirectHostViews {
    fun floats(handle: FfiCpuTensor, keepAlive: AutoCloseable, closeKeepAlive: Boolean = false): FloatTensorView {
        requireDtype(handle, DataType.F32)
        val buf = mapBytes(handle)
        return DirectFloatView(keepAlive, closeKeepAlive, buf.asFloatBuffer())
    }

    fun ints(handle: FfiCpuTensor, keepAlive: AutoCloseable, closeKeepAlive: Boolean = false): IntTensorView {
        requireDtype(handle, DataType.I32)
        val buf = mapBytes(handle)
        return DirectIntView(keepAlive, closeKeepAlive, buf.asIntBuffer())
    }

    fun longs(handle: FfiCpuTensor, keepAlive: AutoCloseable, closeKeepAlive: Boolean = false): LongTensorView {
        requireDtype(handle, DataType.I64)
        val buf = mapBytes(handle)
        return DirectLongView(keepAlive, closeKeepAlive, buf.asLongBuffer())
    }

    fun bytes(handle: FfiCpuTensor, keepAlive: AutoCloseable, closeKeepAlive: Boolean = false): ByteTensorView {
        val buf = mapBytes(handle)
        return DirectByteView(keepAlive, closeKeepAlive, buf)
    }

    private fun mapBytes(handle: FfiCpuTensor): ByteBuffer {
        val host = withFfiErrors { handle.hostDataPtr() }
        return HostBufferBridge.wrap(host.ptr.toLong(), host.nbytes.toLong())
    }

    private fun requireDtype(handle: FfiCpuTensor, expected: DataType) {
        val actual = handle.dtype().fromFfi()
        if (actual != expected) {
            throw InfersException.UnsupportedType("expected $expected tensor, got $actual")
        }
    }
}

internal class DirectFloatView(
    private val keepAlive: AutoCloseable,
    private val closeKeepAlive: Boolean,
    private val buffer: FloatBuffer,
) : FloatTensorView {
    private val gate = CloseGate("FloatTensorView")

    override val size: Int = buffer.remaining()

    override fun get(index: Int): Float {
        gate.ensureOpen()
        require(index in 0..<size) { "index $index out of bounds for size $size" }
        return buffer.get(index)
    }

    override fun copyInto(destination: FloatArray, destinationOffset: Int, startIndex: Int, endIndex: Int) {
        gate.ensureOpen()
        checkViewRange(size, destination.size, destinationOffset, startIndex, endIndex)
        val count = endIndex - startIndex
        if (count == 0) return
        val dup = buffer.duplicate()
        dup.position(startIndex)
        dup.limit(endIndex)
        dup.get(destination, destinationOffset, count)
    }

    fun asBuffer(): FloatBuffer {
        gate.ensureOpen()
        return buffer.asReadOnlyBuffer()
    }

    override fun close() {
        if (gate.markClosed() && closeKeepAlive) {
            keepAlive.close()
        }
    }
}

internal class DirectIntView(
    private val keepAlive: AutoCloseable,
    private val closeKeepAlive: Boolean,
    private val buffer: IntBuffer,
) : IntTensorView {
    private val gate = CloseGate("IntTensorView")

    override val size: Int = buffer.remaining()

    override fun get(index: Int): Int {
        gate.ensureOpen()
        require(index in 0..<size) { "index $index out of bounds for size $size" }
        return buffer.get(index)
    }

    override fun copyInto(destination: IntArray, destinationOffset: Int, startIndex: Int, endIndex: Int) {
        gate.ensureOpen()
        checkViewRange(size, destination.size, destinationOffset, startIndex, endIndex)
        val count = endIndex - startIndex
        if (count == 0) return
        val dup = buffer.duplicate()
        dup.position(startIndex)
        dup.limit(endIndex)
        dup.get(destination, destinationOffset, count)
    }

    fun asBuffer(): IntBuffer {
        gate.ensureOpen()
        return buffer.asReadOnlyBuffer()
    }

    override fun close() {
        if (gate.markClosed() && closeKeepAlive) {
            keepAlive.close()
        }
    }
}

internal class DirectLongView(
    private val keepAlive: AutoCloseable,
    private val closeKeepAlive: Boolean,
    private val buffer: LongBuffer,
) : LongTensorView {
    private val gate = CloseGate("LongTensorView")

    override val size: Int = buffer.remaining()

    override fun get(index: Int): Long {
        gate.ensureOpen()
        require(index in 0..<size) { "index $index out of bounds for size $size" }
        return buffer.get(index)
    }

    override fun copyInto(destination: LongArray, destinationOffset: Int, startIndex: Int, endIndex: Int) {
        gate.ensureOpen()
        checkViewRange(size, destination.size, destinationOffset, startIndex, endIndex)
        val count = endIndex - startIndex
        if (count == 0) return
        val dup = buffer.duplicate()
        dup.position(startIndex)
        dup.limit(endIndex)
        dup.get(destination, destinationOffset, count)
    }

    fun asBuffer(): LongBuffer {
        gate.ensureOpen()
        return buffer.asReadOnlyBuffer()
    }

    override fun close() {
        if (gate.markClosed() && closeKeepAlive) {
            keepAlive.close()
        }
    }
}

internal class DirectByteView(
    private val keepAlive: AutoCloseable,
    private val closeKeepAlive: Boolean,
    private val buffer: ByteBuffer,
) : ByteTensorView {
    private val gate = CloseGate("ByteTensorView")

    override val size: Int = buffer.remaining()

    override fun get(index: Int): Byte {
        gate.ensureOpen()
        require(index in 0..<size) { "index $index out of bounds for size $size" }
        return buffer.get(index)
    }

    override fun copyInto(destination: ByteArray, destinationOffset: Int, startIndex: Int, endIndex: Int) {
        gate.ensureOpen()
        checkViewRange(size, destination.size, destinationOffset, startIndex, endIndex)
        val count = endIndex - startIndex
        if (count == 0) return
        val dup = buffer.duplicate()
        dup.position(startIndex)
        dup.limit(endIndex)
        dup.get(destination, destinationOffset, count)
    }

    fun asBuffer(): ByteBuffer {
        gate.ensureOpen()
        return buffer.asReadOnlyBuffer()
    }

    override fun close() {
        if (gate.markClosed() && closeKeepAlive) {
            keepAlive.close()
        }
    }
}
