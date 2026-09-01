//! Primary crate for the Infers on-device inference runtime.
//!
//! Re-exports [`infers_core`], [`processing`], and [`infers_backend_executorch`] so
//! applications can depend on a single package. Enable `vulkan` for GPU preprocessing
//! and the ExecuTorch Vulkan delegate.

pub use infers_core::*;
pub use processing::CpuImageProcessor;
pub use processing::{
    convert_layout_cpu, infer_rgb_layout, nhwc_to_nchw_cpu,
};
#[cfg(feature = "vulkan")]
pub use processing::{
    convert_layout_vulkan, GpuImageProcessor, LayoutGpuPass, nhwc_to_nchw_gpu,
};
pub use infers_backend_executorch::{
    config::{VulkanOptions, XnnpackOptions},
    ExecuTorchBackend, ExecuTorchDelegate, ExecuTorchSession, ProgramMetadata, TensorDescriptor,
};
#[cfg(feature = "vulkan")]
pub use infers_gpu::{GpuError, Vulkan, VulkanContext, VulkanImage};

#[cfg(target_os = "android")]
pub use platform_android::{
    AndroidHardwareBufferHandle, AndroidPlatformError, LockedCpuBuffer, create_vulkan_context,
    vulkan_context_options,
};
