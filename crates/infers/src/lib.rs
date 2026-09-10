//! Primary crate for the on-device inference runtime.
//!
//! Re-exports [`infers_core`], [`infers_processing`], and [`infers_backend_executorch`] so
//! applications can depend on a single package. Enable `vulkan` for GPU preprocessing
//! and the ExecuTorch Vulkan delegate.

pub use infers_core::*;
pub use infers_processing::CpuImageProcessor;
pub use infers_processing::ImageProcessor;
#[cfg(feature = "vulkan")]
pub use infers_processing::DeferredVulkanProcessExt;
pub use infers_processing::DeferredCpuProcessExt;
pub use infers_backend_executorch::{
    config::{VulkanOptions, XnnpackOptions},
    ExecuTorchBackend, ExecuTorchDelegate, ExecuTorchSession, ProgramMetadata, TensorDescriptor,
};
#[cfg(feature = "vulkan")]
pub use infers_processing::GpuImageProcessor;
#[cfg(feature = "vulkan")]
pub use infers_gpu::{defer_hardware, GpuError, Vulkan, VulkanBufferHandle, VulkanContext, VulkanImage};

/// GPU placement for [`HardwareImage`].
#[cfg(feature = "vulkan")]
pub trait HardwareImageVulkanExt {
    fn on(self, device: &Vulkan) -> Deferred<Vulkan>;
}

#[cfg(feature = "vulkan")]
impl HardwareImageVulkanExt for HardwareImage {
    fn on(self, device: &Vulkan) -> Deferred<Vulkan> {
        defer_hardware(device, self)
    }
}

#[cfg(target_os = "android")]
pub use infers_platform_android::{
    AndroidHardwareBufferHandle, AndroidPlatformError, LockedCpuBuffer, create_vulkan_context,
    vulkan_context_options,
};
