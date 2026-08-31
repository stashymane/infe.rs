pub mod buffer;
pub mod layout_pass;
pub mod processor;
pub mod transfer;

pub use buffer::GpuTensorBuffer;
pub use layout_pass::{LayoutGpuPass, nhwc_to_nchw_gpu};
pub use processor::GpuImageProcessor;
pub use transfer::{upload_tensor_buffer, VulkanDeviceTransfer};

pub(crate) const SHADERS: &[u8] = include_bytes!(env!("PROCESSING_SHADERS_SPV"));
