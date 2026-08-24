use infers_core::CoreError;
use processing::{GpuError, ProcessingError};

#[cfg(target_os = "android")]
use platform_android::AndroidPlatformError;

#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum InfersError {
    #[error("Invalid tensor shape: {message}")]
    InvalidShape { message: String },

    #[error("Device mismatch: expected {expected}, actual {actual}")]
    DeviceMismatch { expected: String, actual: String },

    #[error("Unsupported data type: {message}")]
    UnsupportedType { message: String },

    #[error("Buffer read failed: {message}")]
    BufferReadFailed { message: String },

    #[error("Buffer allocation failed: {message}")]
    BufferAllocationFailed { message: String },

    #[error("Model load failed: {message}")]
    ModelLoadFailed { message: String },

    #[error("Inference execution failed: {message}")]
    InferenceFailed { message: String },

    #[error("Image processing failed: {message}")]
    ProcessingFailed { message: String },

    #[error("Platform error: {message}")]
    PlatformError { message: String },

    #[error("Internal error: {message}")]
    InternalError { message: String },
}

impl From<CoreError> for InfersError {
    fn from(err: CoreError) -> Self {
        match err {
            CoreError::InvalidShape(msg) => InfersError::InvalidShape { message: msg },
            CoreError::DeviceMismatch { expected, actual } => InfersError::DeviceMismatch {
                expected: expected.name,
                actual: actual.name,
            },
            CoreError::InvalidDataType { expected, actual } => InfersError::UnsupportedType {
                message: format!("Expected {expected:?}, got {actual:?}"),
            },
            CoreError::BufferTransferFailed(msg) => InfersError::BufferAllocationFailed { message: msg },
            CoreError::ModelLoadFailed(msg) => InfersError::ModelLoadFailed { message: msg },
            CoreError::InferenceFailed(msg) => InfersError::InferenceFailed { message: msg },
            CoreError::InvalidArgument(msg) => InfersError::InvalidShape { message: msg },
            CoreError::PlatformError(msg) => InfersError::PlatformError { message: msg },
            CoreError::Other(msg) => InfersError::InternalError { message: msg },
        }
    }
}

impl From<ProcessingError> for InfersError {
    fn from(err: ProcessingError) -> Self {
        InfersError::ProcessingFailed {
            message: err.to_string(),
        }
    }
}

impl From<GpuError> for InfersError {
    fn from(err: GpuError) -> Self {
        ProcessingError::from(err).into()
    }
}

#[cfg(target_os = "android")]
impl From<AndroidPlatformError> for InfersError {
    fn from(err: AndroidPlatformError) -> Self {
        InfersError::PlatformError {
            message: err.to_string(),
        }
    }
}
