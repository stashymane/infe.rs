//! Vulkan image preprocessing and GPU tensor buffers built on a shared [`VulkanContext`].

pub mod cpu;

#[cfg(feature = "vulkan")]
pub mod gpu;

pub use cpu::CpuImageProcessor;
pub use processing_core::*;

#[cfg(feature = "vulkan")]
pub use gpu::{GpuImageProcessor, GpuTensorBuffer};

#[cfg(feature = "vulkan")]
pub use infers_gpu::{GpuError, VulkanContext, VulkanSampledImage};
