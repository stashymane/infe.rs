package dev.stashy.infers.internal

import dev.stashy.infers.InfersInternalApi

/** Little-endian codecs for UniFFI byte-buffer tensor transport. */
@InfersInternalApi
public object TensorCodec {
    public fun encodeFloats(data: FloatArray): ByteArray {
        val out = ByteArray(data.size * 4)
        var o = 0
        for (v in data) {
            val bits = v.toRawBits()
            out[o++] = (bits and 0xff).toByte()
            out[o++] = ((bits ushr 8) and 0xff).toByte()
            out[o++] = ((bits ushr 16) and 0xff).toByte()
            out[o++] = ((bits ushr 24) and 0xff).toByte()
        }
        return out
    }

    public fun decodeFloats(bytes: ByteArray): FloatArray {
        require(bytes.size % 4 == 0) { "float byte length ${bytes.size} is not a multiple of 4" }
        val out = FloatArray(bytes.size / 4)
        var i = 0
        var o = 0
        while (i < bytes.size) {
            val bits =
                (bytes[i].toInt() and 0xff) or
                    ((bytes[i + 1].toInt() and 0xff) shl 8) or
                    ((bytes[i + 2].toInt() and 0xff) shl 16) or
                    ((bytes[i + 3].toInt() and 0xff) shl 24)
            out[o++] = Float.fromBits(bits)
            i += 4
        }
        return out
    }

    public fun encodeInts(data: IntArray): ByteArray {
        val out = ByteArray(data.size * 4)
        var o = 0
        for (v in data) {
            out[o++] = (v and 0xff).toByte()
            out[o++] = ((v ushr 8) and 0xff).toByte()
            out[o++] = ((v ushr 16) and 0xff).toByte()
            out[o++] = ((v ushr 24) and 0xff).toByte()
        }
        return out
    }

    public fun decodeInts(bytes: ByteArray): IntArray {
        require(bytes.size % 4 == 0) { "int byte length ${bytes.size} is not a multiple of 4" }
        val out = IntArray(bytes.size / 4)
        var i = 0
        var o = 0
        while (i < bytes.size) {
            out[o++] = (bytes[i].toInt() and 0xff) or
                ((bytes[i + 1].toInt() and 0xff) shl 8) or
                ((bytes[i + 2].toInt() and 0xff) shl 16) or
                ((bytes[i + 3].toInt() and 0xff) shl 24)
            i += 4
        }
        return out
    }

    public fun encodeLongs(data: LongArray): ByteArray {
        val out = ByteArray(data.size * 8)
        var o = 0
        for (v in data) {
            var x = v
            repeat(8) {
                out[o++] = (x and 0xffL).toByte()
                x = x ushr 8
            }
        }
        return out
    }

    public fun decodeLongs(bytes: ByteArray): LongArray {
        require(bytes.size % 8 == 0) { "long byte length ${bytes.size} is not a multiple of 8" }
        val out = LongArray(bytes.size / 8)
        var i = 0
        var o = 0
        while (i < bytes.size) {
            var v = 0L
            for (s in 0 until 8) {
                v = v or ((bytes[i + s].toLong() and 0xffL) shl (s * 8))
            }
            out[o++] = v
            i += 8
        }
        return out
    }
}
