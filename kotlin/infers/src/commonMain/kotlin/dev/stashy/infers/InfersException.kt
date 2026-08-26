package dev.stashy.infers

public sealed class InfersException(
    message: String,
    cause: Throwable? = null,
) : Exception(message, cause) {
    public abstract val reason: String

    public class InvalidShape(
        override val reason: String,
    ) : InfersException("Invalid tensor shape: $reason")

    public class DeviceMismatch(
        public val expected: String,
        public val actual: String,
    ) : InfersException("Device mismatch: expected $expected, actual $actual") {
        override val reason: String
            get() = "expected $expected, actual $actual"
    }

    public class UnsupportedType(
        override val reason: String,
    ) : InfersException("Unsupported data type: $reason")

    public class BufferAllocationFailed(
        override val reason: String,
    ) : InfersException("Buffer allocation failed: $reason")

    public class ModelLoadFailed(
        override val reason: String,
    ) : InfersException("Model load failed: $reason")

    public class InferenceFailed(
        override val reason: String,
    ) : InfersException("Inference execution failed: $reason")

    public class ProcessingFailed(
        override val reason: String,
    ) : InfersException("Image processing failed: $reason")

    public class PlatformError(
        override val reason: String,
    ) : InfersException("Platform error: $reason")

    public class InternalError(
        override val reason: String,
    ) : InfersException("Internal error: $reason")
}
