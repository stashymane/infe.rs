pub mod backend;
pub mod config;
pub mod delegate;
pub mod error;
pub mod program;
pub mod session;
pub mod tensor;
#[cfg(feature = "vulkan")]
pub mod gpu_input;
#[cfg(feature = "vulkan")]
pub mod vulkan_adapter;
#[cfg(feature = "vulkan")]
pub use vulkan_adapter::{register_external_adapter, ExternalAdapterRegistration};

// Re-export native executorch modules and types
pub use executorch::data_loader::{BufferDataLoader, DataLoader};
pub use executorch::module::{LoadMode, Module, ModuleBuilder};
pub use executorch::program::{
    MethodMeta, Program as NativeProgram, ProgramVerification, TensorInfo,
};
pub use executorch::tensor::{ScalarType, Tensor, TensorImpl};

// Re-export backend abstractions
pub use backend::ExecuTorchBackend;
pub use config::ExecuTorchBackendConfig;
pub use delegate::ExecuTorchDelegate;
pub use error::ExecuTorchError;
pub use program::{MethodDescriptor, ProgramMetadata, TensorDescriptor};
pub use session::ExecuTorchSession;
pub use tensor::{
    data_type_to_scalar_type, scalar_type_to_data_type, ExecuTorchTensorBuffer,
};
