//! Vulkan image preprocessing and GPU tensor buffers built on a shared [`VulkanContext`].

pub mod cpu;
pub mod layout;

#[cfg(feature = "vulkan")]
pub mod gpu;

pub use cpu::CpuImageProcessor;
pub use layout::{
    convert_layout, infer_rgb_layout, model_input_from_preprocess, nhwc_to_nchw_cpu,
};
pub use processing_core::*;

#[cfg(feature = "vulkan")]
pub use gpu::{
    GpuImageProcessor, GpuTensorBuffer, LayoutGpuPass, VulkanDeviceTransfer, nhwc_to_nchw_gpu,
    upload_tensor_buffer,
};

#[cfg(feature = "vulkan")]
pub use infers_gpu::{GpuError, VulkanContext, VulkanSampledImage};
