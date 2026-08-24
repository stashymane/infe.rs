pub mod buffer;
pub mod error;
pub mod processor;

pub use buffer::GpuTensorBuffer;
pub use error::GpuError;
pub use processor::GpuImageProcessor;

pub const SHADERS: &[u8] = include_bytes!(env!("PROCESSING_SHADERS_SPV"));
