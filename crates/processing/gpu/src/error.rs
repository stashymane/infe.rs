use infers_core::CoreError;
use infers_gpu::GpuContextError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GpuError {
    #[error("Invalid input image buffer: {0}")]
    InvalidBuffer(String),

    #[error("GPU error: {0}")]
    Gpu(String),

    #[error("Core error: {0}")]
    Core(#[from] CoreError),
}

impl From<GpuContextError> for GpuError {
    fn from(err: GpuContextError) -> Self {
        GpuError::Gpu(err.to_string())
    }
}

impl From<ash::vk::Result> for GpuError {
    fn from(err: ash::vk::Result) -> Self {
        GpuError::Gpu(err.to_string())
    }
}
