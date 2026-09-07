package dev.stashy.infers

import dev.stashy.infers.internal.DirectByteView
import dev.stashy.infers.internal.DirectFloatView
import dev.stashy.infers.internal.DirectIntView
import dev.stashy.infers.internal.DirectLongView
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer

/** Zero-copy little-endian view of host float storage. Invalid after the view or tensor is closed. */
public fun FloatTensorView.asBuffer(): FloatBuffer = (this as? DirectFloatView)?.asBuffer()
    ?: error("asBuffer requires a DirectByteBuffer-backed float view")

/** Zero-copy little-endian view of host int storage. Invalid after the view or tensor is closed. */
public fun IntTensorView.asBuffer(): IntBuffer = (this as? DirectIntView)?.asBuffer()
    ?: error("asBuffer requires a DirectByteBuffer-backed int view")

/** Zero-copy little-endian view of host long storage. Invalid after the view or tensor is closed. */
public fun LongTensorView.asBuffer(): LongBuffer = (this as? DirectLongView)?.asBuffer()
    ?: error("asBuffer requires a DirectByteBuffer-backed long view")

/** Zero-copy view of host byte storage. Invalid after the view or tensor is closed. */
public fun ByteTensorView.asBuffer(): ByteBuffer = (this as? DirectByteView)?.asBuffer()
    ?: error("asBuffer requires a DirectByteBuffer-backed byte view")
