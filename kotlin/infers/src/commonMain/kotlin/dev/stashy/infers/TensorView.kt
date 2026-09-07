package dev.stashy.infers

/**
 * Typed view over host tensor storage. Use [copyInto] for owned slices or [asBuffer] for
 * zero-copy access on JVM/Android.
 *
 * Typed views are [Iterable] for sequential scans (`for (v in view)`). Prefer [copyInto] /
 * [asBuffer] for bulk access — avoid materializing with `toList()` / similar.
 *
 * Views remain valid while open and while the underlying tensor is open.
 */
public interface TensorView : AutoCloseable {
    public val size: Int
}

/** Little-endian float view of an F32 tensor. */
public interface FloatTensorView :
    TensorView,
    Iterable<Float> {
    public operator fun get(index: Int): Float

    public fun copyInto(destination: FloatArray, destinationOffset: Int = 0, startIndex: Int = 0, endIndex: Int = size)

    override fun iterator(): Iterator<Float> = IndexIterator(size) { get(it) }
}

/** Little-endian int view of an I32 tensor. */
public interface IntTensorView :
    TensorView,
    Iterable<Int> {
    public operator fun get(index: Int): Int

    public fun copyInto(destination: IntArray, destinationOffset: Int = 0, startIndex: Int = 0, endIndex: Int = size)

    override fun iterator(): Iterator<Int> = IndexIterator(size) { get(it) }
}

/** Little-endian long view of an I64 tensor. */
public interface LongTensorView :
    TensorView,
    Iterable<Long> {
    public operator fun get(index: Int): Long

    public fun copyInto(destination: LongArray, destinationOffset: Int = 0, startIndex: Int = 0, endIndex: Int = size)

    override fun iterator(): Iterator<Long> = IndexIterator(size) { get(it) }
}

/** Byte view of raw tensor storage. */
public interface ByteTensorView :
    TensorView,
    Iterable<Byte> {
    public operator fun get(index: Int): Byte

    public fun copyInto(destination: ByteArray, destinationOffset: Int = 0, startIndex: Int = 0, endIndex: Int = size)

    override fun iterator(): Iterator<Byte> = IndexIterator(size) { get(it) }
}

private class IndexIterator<T>(
    private val size: Int,
    private val get: (Int) -> T,
) : Iterator<T> {
    private var index = 0

    override fun hasNext(): Boolean = index < size

    override fun next(): T {
        if (!hasNext()) throw NoSuchElementException()
        return get(index++)
    }
}

internal fun checkViewRange(size: Int, destinationSize: Int, destinationOffset: Int, startIndex: Int, endIndex: Int) {
    require(startIndex in 0..size) { "startIndex $startIndex out of bounds for size $size" }
    require(endIndex in startIndex..size) { "endIndex $endIndex out of bounds for size $size" }
    val count = endIndex - startIndex
    require(destinationOffset >= 0) { "destinationOffset $destinationOffset < 0" }
    require(destinationOffset + count <= destinationSize) {
        "destination cannot hold $count elements at offset $destinationOffset (capacity $destinationSize)"
    }
}
