package dev.stashy.infers

import dev.stashy.infers.ffi.createHardwareImage
import dev.stashy.infers.internal.CloseGate
import dev.stashy.infers.internal.fromFfi
import dev.stashy.infers.internal.toFfi
import dev.stashy.infers.internal.withFfiErrors
import dev.stashy.infers.ffi.HardwareImage as FfiHardwareImage

/** Host-resident image bytes (camera frame, decoded file, etc.). */
public class HardwareImage @InfersInternalApi constructor(
    @InfersInternalApi
    public val handle: FfiHardwareImage,
) : AutoCloseable {
    private val gate = CloseGate("HardwareImage")

    public val width: UInt
        get() {
            gate.ensureOpen()
            return handle.width()
        }

    public val height: UInt
        get() {
            gate.ensureOpen()
            return handle.height()
        }

    public val format: ImageFormat
        get() {
            gate.ensureOpen()
            return handle.format().fromFfi()
        }

    @InfersInternalApi
    public fun ensureOpen(): Unit = gate.ensureOpen()

    override fun close() {
        if (gate.markClosed()) {
            handle.close()
        }
    }

    public companion object {
        @InfersInternalApi
        internal fun fromBytesInternal(
            bytes: ByteArray,
            width: UInt,
            height: UInt,
            format: ImageFormat,
        ): HardwareImage = withFfiErrors {
            HardwareImage(createHardwareImage(width, height, format.toFfi(), bytes))
        }

        @OptIn(InfersInternalApi::class)
        context(scope: InferenceScope)
        public fun fromBytes(bytes: ByteArray, width: UInt, height: UInt, format: ImageFormat): HardwareImage =
            scope.register(fromBytesInternal(bytes, width, height, format))
    }
}

@OptIn(InfersInternalApi::class)
context(scope: InferenceScope)
public fun HardwareImage.onCpu(): CpuDeferred = scope.register(CpuDeferred(handle.onCpu()))
