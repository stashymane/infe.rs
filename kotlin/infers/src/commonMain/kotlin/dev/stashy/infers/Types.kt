package dev.stashy.infers

public enum class DataType {
    U8,
    I8,
    I16,
    I32,
    I64,
    F16,
    F32,
    F64,
}

public enum class ImageFormat {
    Rgb888,
    Rgbf32,
    Nv12,
    I420,
}

public enum class FitMode {
    Stretch,
    Contain,
    Crop,
}

public enum class Rotation {
    None,
    Rot90,
    Rot180,
    Rot270,
}

public enum class TensorLayout {
    Nhwc,
    Nchw,
}

public data class TensorShape(
    val dims: List<ULong>,
) {
    public companion object {
        public fun of(vararg dims: Long): TensorShape = TensorShape(dims.map { it.toULong() })
    }
}

public data class ProcessingOptions(
    public val srcW: UInt,
    public val srcH: UInt,
    public val cropX: UInt,
    public val cropY: UInt,
    public val cropW: UInt,
    public val cropH: UInt,
    public val destW: UInt,
    public val destH: UInt,
    public val srcFormat: ImageFormat,
    public val destFormat: ImageFormat,
    public val fitMode: FitMode = FitMode.Stretch,
    public val rotation: Rotation = Rotation.None,
    public val destLayout: TensorLayout = destFormat.defaultLayout(),
) {
    public companion object {
        /**
         * Builds options with a DSL:
         * ```
         * ProcessingOptions {
         *     source = 1920 to 1080
         *     dest = 300 to 300
         *     srcFormat = ImageFormat.Rgb888
         *     destFormat = ImageFormat.Rgbf32
         * }
         * ```
         * Crop defaults to the full source frame when [ProcessingOptionsBuilder.cropSize] is unset.
         */
        public operator fun invoke(block: ProcessingOptionsBuilder.() -> Unit): ProcessingOptions =
            ProcessingOptionsBuilder().apply(block).build()
    }
}

/** DSL marker for [ProcessingOptionsBuilder]. */
@DslMarker
public annotation class ProcessingOptionsDsl

/**
 * Builder for [ProcessingOptions]. Dimensions use `width to height` pairs
 * (e.g. `source = 1920 to 1080`).
 */
@ProcessingOptionsDsl
public class ProcessingOptionsBuilder {
    /** Source image size as `width to height`. Required. */
    public var source: Pair<Int, Int>? = null

    /** Destination size as `width to height`. Required. */
    public var dest: Pair<Int, Int>? = null

    /** Crop top-left as `x to y`. Defaults to `0 to 0`. */
    public var cropOrigin: Pair<Int, Int> = 0 to 0

    /**
     * Crop size as `width to height`. When null, the crop covers the full
     * [source] frame.
     */
    public var cropSize: Pair<Int, Int>? = null

    /** Required source pixel format. */
    public var srcFormat: ImageFormat? = null

    /** Required destination pixel format. */
    public var destFormat: ImageFormat? = null

    public var fitMode: FitMode = FitMode.Stretch

    public var rotation: Rotation = Rotation.None

    public var destLayout: TensorLayout? = null

    public fun build(): ProcessingOptions {
        val source =
            requireNotNull(source) { "source (width to height) is required" }
        val dest =
            requireNotNull(dest) { "dest (width to height) is required" }
        val srcFormat =
            requireNotNull(srcFormat) { "srcFormat is required" }
        val destFormat =
            requireNotNull(destFormat) { "destFormat is required" }

        val (srcW, srcH) = source.toNonNegativeUIntPair("source")
        val (destW, destH) = dest.toNonNegativeUIntPair("dest")
        val (cropX, cropY) = cropOrigin.toNonNegativeUIntPair("cropOrigin")
        val (cropW, cropH) = (cropSize ?: source).toNonNegativeUIntPair("cropSize")

        return ProcessingOptions(
            srcW = srcW,
            srcH = srcH,
            cropX = cropX,
            cropY = cropY,
            cropW = cropW,
            cropH = cropH,
            destW = destW,
            destH = destH,
            srcFormat = srcFormat,
            destFormat = destFormat,
            fitMode = fitMode,
            rotation = rotation,
            destLayout = destLayout ?: destFormat.defaultLayout(),
        )
    }
}

private fun ImageFormat.defaultLayout(): TensorLayout = when (this) {
    ImageFormat.Rgbf32 -> TensorLayout.Nchw
    ImageFormat.Rgb888, ImageFormat.Nv12, ImageFormat.I420 -> TensorLayout.Nhwc
}

private fun Pair<Int, Int>.toNonNegativeUIntPair(label: String): Pair<UInt, UInt> {
    require(first >= 0 && second >= 0) {
        "$label dimensions must be non-negative, got $first to $second"
    }
    return first.toUInt() to second.toUInt()
}
