use infers_core::Device;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GpuContextError {
    #[error("Vulkan loader failed: {0}")]
    Loader(String),

    #[error("Vulkan error: {0}")]
    Vulkan(#[from] ash::vk::Result),

    #[error("No compute-capable GPU at index {0}")]
    NoDevice(usize),

    #[error("Cannot create Vulkan context for non-GPU device {0:?}")]
    NotGpu(Device),

    #[error("Missing required Vulkan feature or extension: {0}")]
    MissingFeature(String),

    #[error("GPU allocator error: {0}")]
    Allocator(String),

    #[error("{0}")]
    Other(String),
}

impl From<gpu_allocator::AllocationError> for GpuContextError {
    fn from(err: gpu_allocator::AllocationError) -> Self {
        GpuContextError::Allocator(err.to_string())
    }
}
