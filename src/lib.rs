//! Primary crate for the Infers on-device inference runtime.
//!
//! Re-exports the workspace libraries used to build and run pipelines:
//! core types and traits, image preprocessing, the ExecuTorch backend, and
//! Android hardware-buffer support when targeting Android.

/// Core types, tensors, devices, and backend traits.
pub mod core {
    pub use infers_core::*;
}

/// CPU and GPU image preprocessing.
pub mod processing {
    pub use ::processing::*;
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
    reference_shader_convert_main,
};
pub use executorch::{
    ExecuTorchBackend, ExecuTorchDelegate, ExecuTorchError, ExecuTorchSession, ExecuTorchTensorBuffer,
    MethodDescriptor, ProgramMetadata, TensorDescriptor, data_type_to_scalar_type,
    scalar_type_to_data_type,
};

#[cfg(target_os = "android")]
pub use android::{
    AndroidHardwareBufferHandle, AndroidPlatformError, LockedCpuBuffer, VulkanHardwareBufferImport,
};
