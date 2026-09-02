use infers_core::CoreError;

#[cfg(feature = "vulkan")]
use infers_gpu::GpuError;

#[cfg(target_os = "android")]
use platform_android::AndroidPlatformError;

/// UniFFI error type. Field is named `reason` (not `message`) so generated Kotlin
/// does not clash with `Throwable.message`.
#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum InfersError {
    #[error("Invalid tensor shape: {reason}")]
    InvalidShape { reason: String },

    #[error("Device mismatch: expected {expected}, actual {actual}")]
    DeviceMismatch { expected: String, actual: String },

    #[error("Unsupported data type: {reason}")]
    UnsupportedType { reason: String },

    #[error("Buffer allocation failed: {reason}")]
    BufferAllocationFailed { reason: String },

    #[error("Model load failed: {reason}")]
    ModelLoadFailed { reason: String },

    #[error("Inference execution failed: {reason}")]
    InferenceFailed { reason: String },

    #[error("Image processing failed: {reason}")]
    ProcessingFailed { reason: String },

    #[error("Platform error: {reason}")]
    PlatformError { reason: String },

    #[error("Internal error: {reason}")]
    InternalError { reason: String },

    #[error("Handle already consumed")]
    AlreadyConsumed,
}

impl From<CoreError> for InfersError {
    fn from(err: CoreError) -> Self {
        match err {
            CoreError::InvalidShape(msg) => InfersError::InvalidShape { reason: msg },
            CoreError::DeviceMismatch { expected, actual } => InfersError::DeviceMismatch {
                expected: expected.name,
                actual: actual.name,
            },
            CoreError::InvalidDataType { expected, actual } => InfersError::UnsupportedType {
                reason: format!("Expected {expected:?}, got {actual:?}"),
            },
            CoreError::BufferTransferFailed(msg) => {
                InfersError::BufferAllocationFailed { reason: msg }
            }
            CoreError::ModelLoadFailed(msg) => InfersError::ModelLoadFailed { reason: msg },
            CoreError::InferenceFailed(msg) => InfersError::InferenceFailed { reason: msg },
            CoreError::InvalidArgument(msg) => InfersError::InvalidShape { reason: msg },
            CoreError::InvalidImageBuffer(msg) | CoreError::ImageResizeFailed(msg) => {
                InfersError::ProcessingFailed { reason: msg }
            }
            CoreError::Platform(msg) => InfersError::PlatformError { reason: msg },
            CoreError::Gpu(err) => InfersError::ProcessingFailed {
                reason: err.to_string(),
            },
            CoreError::AlreadyConsumed => InfersError::AlreadyConsumed,
        }
    }
}

#[cfg(feature = "vulkan")]
impl From<GpuError> for InfersError {
    fn from(err: GpuError) -> Self {
        CoreError::from(err).into()
    }
}

#[cfg(target_os = "android")]
impl From<AndroidPlatformError> for InfersError {
    fn from(err: AndroidPlatformError) -> Self {
        CoreError::from(err).into()
    }
}
