pub mod buffer;
pub mod processor;

pub use buffer::GpuTensorBuffer;
pub use processor::GpuImageProcessor;

pub(crate) const SHADERS: &[u8] = include_bytes!(env!("PROCESSING_SHADERS_SPV"));
