pub mod cpu;
pub mod error;
pub mod gpu;

pub use cpu::{reference_shader_convert_main, CpuImageProcessor, CpuTensorBuffer};
pub use error::ProcessingError;
pub use gpu::{GpuImageProcessor, GpuTensorBuffer};
pub use processing_core as core;
pub use processing_core::*;

pub const SHADERS: &[u8] = include_bytes!(env!("PROCESSING_SHADERS_SPV"));
