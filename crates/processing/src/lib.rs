//! Vulkan image preprocessing built on a shared [`Vulkan`](infers_gpu::Vulkan) device.

pub mod cpu;
pub mod layout;
pub mod processor;

#[cfg(feature = "vulkan")]
pub mod gpu;

pub use cpu::CpuImageProcessor;
pub use processor::{DeferredCpuProcessExt, ImageProcessor};
#[cfg(feature = "vulkan")]
pub use processor::DeferredVulkanProcessExt;
pub use processing_core::*;

#[cfg(feature = "vulkan")]
pub use gpu::GpuImageProcessor;
#[cfg(feature = "vulkan")]
pub use infers_gpu::{
    GpuError, Vulkan, VulkanContext, VulkanImage, VulkanSampledImage, VulkanStorage,
};
