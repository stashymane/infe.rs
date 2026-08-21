use thiserror::Error;
use crate::device::Device;
use crate::tensor::DataType;

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum CoreError {
    #[error("Device mismatch: expected {expected:?}, got {actual:?}")]
    DeviceMismatch { expected: Device, actual: Device },

    #[error("Invalid tensor shape: {0}")]
    InvalidShape(String),

    #[error("Invalid data type: expected {expected:?}, got {actual:?}")]
    InvalidDataType { expected: DataType, actual: DataType },

    #[error("Buffer transfer failed: {0}")]
    BufferTransferFailed(String),

    #[error("Model load failed: {0}")]
    ModelLoadFailed(String),

    #[error("Inference execution failed: {0}")]
    InferenceFailed(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("Platform error: {0}")]
    PlatformError(String),

    #[error("General error: {0}")]
    Other(String),
}
