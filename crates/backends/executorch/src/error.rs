use infers_core::{CoreError, DataType};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ExecuTorchError {
    #[error("Invalid ExecuTorch program format: {0}")]
    InvalidProgram(String),

    #[error("Method '{0}' not found in ExecuTorch program")]
    MethodNotFound(String),

    #[error("Tensor data type mismatch: expected {expected:?}, got {actual:?}")]
    DataTypeMismatch {
        expected: DataType,
        actual: DataType,
    },

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
            ExecuTorchError::DataTypeMismatch { expected, actual } => {
                CoreError::InvalidDataType { expected, actual }
            }
            ExecuTorchError::MethodNotFound(name) => {
                CoreError::ModelLoadFailed(format!("Method '{}' not found", name))
            }
            ExecuTorchError::InvalidProgram(msg) => CoreError::ModelLoadFailed(msg),
            ExecuTorchError::Execution(msg) => CoreError::InferenceFailed(msg),
            ExecuTorchError::BufferError(msg) => CoreError::BufferTransferFailed(msg),
            ExecuTorchError::Native(e) => match e {
                executorch::Error::InvalidProgram
                | executorch::Error::InvalidExternalData
                | executorch::Error::InvalidArgument
                | executorch::Error::AccessFailed
                | executorch::Error::NotFound => {
                    CoreError::ModelLoadFailed(format!("ExecuTorch native error: {:?}", e))
                }
                other => CoreError::InferenceFailed(format!("ExecuTorch native error: {:?}", other)),
            },
            ExecuTorchError::Io(e) => CoreError::ModelLoadFailed(e.to_string()),
            ExecuTorchError::Core(c) => c,
        }
    }
}
