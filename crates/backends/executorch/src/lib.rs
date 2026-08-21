pub mod backend;
pub mod delegate;
pub mod error;
pub mod program;
pub mod session;
pub mod tensor;

// Re-export native executorch modules and types
pub use executorch::data_loader::{BufferDataLoader, DataLoader};
pub use executorch::module::{LoadMode, Module, ModuleBuilder};
pub use executorch::program::{
    MethodMeta, Program as NativeProgram, ProgramVerification, TensorInfo,
};
pub use executorch::tensor::{ScalarType, Tensor, TensorImpl};

// Re-export backend abstractions
pub use backend::ExecuTorchBackend;
pub use delegate::ExecuTorchDelegate;
pub use error::ExecuTorchError;
pub use program::{MethodDescriptor, ProgramMetadata, TensorDescriptor};
pub use session::ExecuTorchSession;
pub use tensor::{
    data_type_to_scalar_type, scalar_type_to_data_type, ExecuTorchTensorBuffer,
};
