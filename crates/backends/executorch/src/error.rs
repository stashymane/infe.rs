use infers_core::{CoreError, DataType, Device};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ExecuTorchError {
    #[error("Invalid ExecuTorch program format: {0}")]
    InvalidProgram(String),

    #[error("Method '{0}' not found in ExecuTorch program")]
    MethodNotFound(String),

    #[error("Device mismatch: expected device {expected:?}, but received input on {actual:?}")]
    DeviceMismatch {
        expected: Device,
        actual: Device,
    },

    #[error("Tensor shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch { expected: String, actual: String },

    #[error("Tensor data type mismatch: expected {expected:?}, got {actual:?}")]
    DataTypeMismatch {
        expected: DataType,
        actual: DataType,
    },

    #[error("Input count mismatch: expected {expected}, got {actual}")]
    InputCountMismatch { expected: usize, actual: usize },

    #[error("Execution error: {0}")]
    Execution(String),

    #[error("Memory allocation or buffer error: {0}")]
    BufferError(String),

    #[error("ExecuTorch runtime error: {0}")]
    Native(#[from] executorch::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Core error: {0}")]
    Core(#[from] CoreError),
}

impl From<ExecuTorchError> for CoreError {
    fn from(err: ExecuTorchError) -> Self {
        match err {
            ExecuTorchError::DeviceMismatch { expected, actual } => {
                CoreError::DeviceMismatch { expected, actual }
            }
            ExecuTorchError::ShapeMismatch { expected, actual } => {
                CoreError::InvalidShape(format!("Expected shape {}, got {}", expected, actual))
            }
            ExecuTorchError::DataTypeMismatch { expected, actual } => {
                CoreError::InvalidDataType { expected, actual }
            }
            ExecuTorchError::InputCountMismatch { expected, actual } => {
                CoreError::InferenceFailed(format!(
                    "Input count mismatch: expected {}, got {}",
                    expected, actual
                ))
            }
            ExecuTorchError::MethodNotFound(name) => {
                CoreError::ModelLoadFailed(format!("Method '{}' not found", name))
            }
            ExecuTorchError::InvalidProgram(msg) => CoreError::ModelLoadFailed(msg),
            ExecuTorchError::Execution(msg) => CoreError::InferenceFailed(msg),
            ExecuTorchError::BufferError(msg) => CoreError::BufferTransferFailed(msg),
            ExecuTorchError::Native(e) => CoreError::InferenceFailed(format!("ExecuTorch native error: {:?}", e)),
            ExecuTorchError::Io(e) => CoreError::ModelLoadFailed(e.to_string()),
            ExecuTorchError::Core(c) => c,
        }
    }
}
