package dev.stashy.infers

import dev.stashy.infers.ffi.createCpuImageProcessor
import dev.stashy.infers.internal.CloseGate
import dev.stashy.infers.internal.fromFfi
import dev.stashy.infers.internal.toFfi
import dev.stashy.infers.internal.withFfiErrors
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import dev.stashy.infers.ffi.ImageProcessor as FfiImageProcessor

public interface ImageProcessor : AutoCloseable {
    public val device: Device
}

/**
 * Shared FFI-backed image processor. Each instance owns a single-threaded
 * dispatcher matching the native mutex, so concurrent calls serialize without
 * parking extra threads.
 */
public open class FfiBackedImageProcessor
    @InfersInternalApi
    public constructor(
        @property:InfersInternalApi
        public val handle: FfiImageProcessor,
        private val ownerName: String,
        @property:InfersInternalApi
        public val dispatcher: CoroutineDispatcher = Dispatchers.Default.limitedParallelism(1),
    ) : ImageProcessor {
        private val gate = CloseGate(ownerName)

        internal fun ensureOpen() {
            gate.ensureOpen()
        }

        override val device: Device
            get() {
                gate.ensureOpen()
                return handle.device().fromFfi()
            }

        /**
         * Preprocesses [bytes] into a tensor.
         *
         * Native calls are not interruptible; cancellation is checked around the call.
         */
        @InfersInternalApi
        public suspend fun processBytes(
            bytes: ByteArray,
            width: UInt,
            height: UInt,
            format: ImageFormat,
            options: ProcessingOptions,
        ): Tensor =
            withContext(dispatcher) {
                withFfiErrors {
                    gate.ensureOpen()
                    Tensor.fromFfi(
                        handle.processBytes(
                            bytes,
                            width,
                            height,
                            format.toFfi(),
                            options.toFfi(),
                        ),
                    )
                }
            }

        @InfersInternalApi
        public suspend fun processBytes(
            bytes: ByteArray,
            options: ProcessingOptions,
        ): Tensor = processBytes(bytes, options.srcW, options.srcH, options.srcFormat, options)

        override fun close() {
            if (gate.markClosed()) {
                handle.close()
            }
        }
    }

public class CpuImageProcessor : FfiBackedImageProcessor {
    @InfersInternalApi
    public constructor(handle: FfiImageProcessor) : super(handle, "CpuImageProcessor")

    public constructor() : this(createCpuImageProcessor())
}
