use crate::delegate::ExecuTorchDelegate;
use crate::error::ExecuTorchError;
use crate::tensor::scalar_type_to_data_type;
pub use executorch::data_loader::{BufferDataLoader, DataLoader};
pub use executorch::program::{
    MethodMeta, Program as NativeProgram, ProgramVerification, TensorInfo,
};
use infers_core::{DataType, TensorShape};
use std::collections::HashMap;

/// A descriptor of a tensor with name, shape, and data type.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorDescriptor {
    pub name: String,
    pub shape: TensorShape,
    pub dtype: DataType,
}

impl TensorDescriptor {
    pub fn new(name: impl Into<String>, shape: TensorShape, dtype: DataType) -> Self {
        Self {
            name: name.into(),
            shape,
            dtype,
        }
    }

    /// Extract a TensorDescriptor from native ExecuTorch TensorInfo metadata.
    pub fn from_tensor_info(name: impl Into<String>, info: &TensorInfo<'_>) -> Result<Self, ExecuTorchError> {
        let dims: Vec<usize> = info.sizes().iter().map(|&s| s as usize).collect();
        let shape = TensorShape::new(dims)?;
        let dtype = scalar_type_to_data_type(info.scalar_type())?;
        Ok(Self::new(name, shape, dtype))
    }
}

/// Metadata and configuration for a method within an ExecuTorch program.
#[derive(Clone, Debug, PartialEq)]
pub struct MethodDescriptor {
    pub name: String,
    pub inputs: Vec<TensorDescriptor>,
    pub outputs: Vec<TensorDescriptor>,
    pub delegate: Option<ExecuTorchDelegate>,
}

impl MethodDescriptor {
    pub fn new(
        name: impl Into<String>,
        inputs: Vec<TensorDescriptor>,
        outputs: Vec<TensorDescriptor>,
    ) -> Self {
        Self {
            name: name.into(),
            inputs,
            outputs,
            delegate: None,
        }
    }

    /// Extract method metadata from native ExecuTorch `MethodMeta`.
    pub fn from_method_meta(meta: &MethodMeta<'_>) -> Result<Self, ExecuTorchError> {
        let name = meta.name().to_string();

        let mut inputs = Vec::new();
        let num_inputs = meta.num_inputs();
        for i in 0..num_inputs {
            if let Ok(info) = meta.input_tensor_meta(i) {
                inputs.push(TensorDescriptor::from_tensor_info(format!("input_{}", i), &info)?);
            }
        }

        let mut outputs = Vec::new();
        let num_outputs = meta.num_outputs();
        for i in 0..num_outputs {
            if let Ok(info) = meta.output_tensor_meta(i) {
                outputs.push(TensorDescriptor::from_tensor_info(format!("output_{}", i), &info)?);
            }
        }

        Ok(Self {
            name,
            inputs,
            outputs,
            delegate: None,
        })
    }
}

/// Metadata representation of an ExecuTorch program.
#[derive(Clone, Debug, Default)]
pub struct ProgramMetadata {
    pub methods: HashMap<String, MethodDescriptor>,
}

impl ProgramMetadata {
    pub fn new() -> Self {
        Self {
            methods: HashMap::new(),
        }
    }

    /// Extract metadata for all methods from a native ExecuTorch Program.
    pub fn from_native_program(program: &NativeProgram<'_>) -> Result<Self, ExecuTorchError> {
        let mut methods = HashMap::new();
        let num_methods = program.num_methods();

        for i in 0..num_methods {
            if let Ok(name) = program.get_method_name(i) {
                if let Ok(c_name) = std::ffi::CString::new(name) {
                    if let Ok(meta) = program.method_meta(&c_name) {
                        if let Ok(method_desc) = MethodDescriptor::from_method_meta(&meta) {
                            methods.insert(name.to_string(), method_desc);
                        }
                    }
                }
            }
        }

        Ok(Self { methods })
    }

    /// Load program metadata from raw model bytes using native ExecuTorch BufferDataLoader and Program.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ExecuTorchError> {
        let loader = BufferDataLoader::new(bytes);
        match NativeProgram::load(&loader, None) {
            Ok(prog) => Self::from_native_program(&prog),
            Err(e) => Err(ExecuTorchError::Native(e)),
        }
    }

    pub fn add_method(&mut self, method: MethodDescriptor) {
        self.methods.insert(method.name.clone(), method);
    }

    pub fn method(&self, name: &str) -> Option<&MethodDescriptor> {
        self.methods.get(name)
    }
}
