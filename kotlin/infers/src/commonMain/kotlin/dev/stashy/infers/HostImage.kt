package dev.stashy.infers

import dev.stashy.infers.ffi.createHostImage
import dev.stashy.infers.internal.CloseGate
import dev.stashy.infers.internal.fromFfi
import dev.stashy.infers.internal.toFfi
import dev.stashy.infers.internal.withFfiErrors
import dev.stashy.infers.ffi.HostImage as FfiHostImage

/** Host-resident image bytes (camera frame, decoded file, etc.). */
public class HostImage @InfersInternalApi constructor(
    @InfersInternalApi
    public val handle: FfiHostImage,
) : DeviceImage<CpuDevice> {
    private val gate = CloseGate("HostImage")

    override val width: UInt
        get() {
            gate.ensureOpen()
            return handle.width()
        }

    override val height: UInt
        get() {
            gate.ensureOpen()
            return handle.height()
        }

    override val format: ImageFormat
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
        internal fun fromBytesInternal(bytes: ByteArray, width: UInt, height: UInt, format: ImageFormat): HostImage =
            withFfiErrors {
                HostImage(createHostImage(width, height, format.toFfi(), bytes))
            }
        
        @OptIn(InfersInternalApi::class)
        context(scope: InferenceScope)
        public fun fromBytes(
            bytes: ByteArray,
            width: UInt,
            height: UInt,
            format: ImageFormat,
        ): DeviceImage<CpuDevice> = scope.register(fromBytesInternal(bytes, width, height, format))
    }
}
