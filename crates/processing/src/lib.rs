//! Vulkan image preprocessing built on a shared [`Vulkan`](infers_gpu::Vulkan) device.

pub mod cpu;
pub mod layout;

#[cfg(feature = "vulkan")]
pub mod gpu;

pub use cpu::CpuImageProcessor;
pub use layout::{convert_layout_cpu, infer_rgb_layout, nhwc_to_nchw_cpu};
#[cfg(feature = "vulkan")]
pub use layout::convert_layout_vulkan;
pub use processing_core::*;

#[cfg(feature = "vulkan")]
pub use gpu::{GpuImageProcessor, LayoutGpuPass, nhwc_to_nchw_gpu};
#[cfg(feature = "vulkan")]
pub use infers_gpu::{
    GpuError, Vulkan, VulkanContext, VulkanImage, VulkanSampledImage, VulkanStorage,
};
