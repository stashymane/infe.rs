//! Primary crate for the Infers on-device inference runtime.
//!
//! Re-exports [`infers_core`], [`processing`], and [`infers_backend_executorch`] so
//! applications can depend on a single package. Enable `vulkan` for GPU preprocessing
//! and the ExecuTorch Vulkan delegate.

pub use infers_core::*;
pub use processing::CpuImageProcessor;
#[cfg(feature = "vulkan")]
pub use processing::{GpuImageProcessor, GpuTensorBuffer};
pub use infers_backend_executorch::{
    ExecuTorchBackend, ExecuTorchBackendConfig, ExecuTorchDelegate, ExecuTorchSession,
    ExecuTorchTensorBuffer, ProgramMetadata, TensorDescriptor,
};
#[cfg(feature = "vulkan")]
pub use infers_gpu::{GpuError, VulkanContext};

#[cfg(target_os = "android")]
pub use platform_android::{
    AndroidHardwareBufferHandle, AndroidPlatformError, LockedCpuBuffer, create_vulkan_context,
    vulkan_context_options,
};
