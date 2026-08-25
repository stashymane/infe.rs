//! Primary crate for the Infers on-device inference runtime.
//!
//! Re-exports the workspace libraries used to build and run pipelines:
//! core types and traits, image preprocessing, the ExecuTorch backend, and
//! Android hardware-buffer support when targeting Android.

/// Core types, tensors, devices, and backend traits.
pub mod core {
    pub use infers_core::*;
}

/// GPU types and a shared Vulkan context (`VulkanContext`).
pub mod gpu {
    pub use infers_gpu::*;
}

/// CPU image preprocessing.
pub mod processing {
    pub use ::processing::*;
}

/// GPU image preprocessing (SPIR-V convert kernels, including YUV sources).
pub mod processing_gpu {
    pub use ::processing_gpu::*;
}

/// ExecuTorch backend, sessions, and native program types.
pub mod executorch {
    pub use infers_backend_executorch::*;
}

/// Android `AHardwareBuffer` integration.
#[cfg(target_os = "android")]
pub mod android {
    pub use platform_android::*;
}

pub use core::*;
pub use processing::{
    CpuImageProcessor, CpuTensorBuffer, GpuImageProcessor, GpuTensorBuffer, ProcessingError, SHADERS,
};
pub use executorch::{
    ExecuTorchBackend, ExecuTorchBackendConfig, ExecuTorchDelegate, ExecuTorchError,
    ExecuTorchSession, ExecuTorchTensorBuffer, MethodDescriptor, ProgramMetadata, TensorDescriptor,
    data_type_to_scalar_type, scalar_type_to_data_type,
};
pub use gpu::VulkanContext;

#[cfg(target_os = "android")]
pub use android::{
    AndroidHardwareBufferHandle, AndroidPlatformError, LockedCpuBuffer, create_vulkan_context,
    vulkan_context_options,
};
