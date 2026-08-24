use infers_core::CoreError;
use processing_gpu::GpuError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ProcessingError {
    #[error("Invalid input image buffer: {0}")]
    InvalidBuffer(String),

    #[error("Resize error: {0}")]
    ResizeError(String),

    #[error("GPU error: {0}")]
    GpuError(String),

    #[error("Platform error: {0}")]
    Platform(String),

    #[error("Core error: {0}")]
    Core(#[from] CoreError),
}

impl From<GpuError> for ProcessingError {
    fn from(err: GpuError) -> Self {
        match err {
            GpuError::InvalidBuffer(msg) => ProcessingError::InvalidBuffer(msg),
            GpuError::Gpu(msg) => ProcessingError::GpuError(msg),
            GpuError::Core(err) => ProcessingError::Core(err),
        }
    }
}
