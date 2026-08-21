use crate::device::Device;
use crate::error::CoreError;
use crate::tensor::{TensorBuffer, TensorShape};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionConfig {
    pub device: Device,
    pub num_threads: usize,
    pub extra_options: std::collections::HashMap<String, String>,
}

impl SessionConfig {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            num_threads: 1,
            extra_options: std::collections::HashMap::new(),
        }
    }

    pub fn with_threads(mut self, threads: usize) -> Self {
        self.num_threads = threads;
        self
    }

    pub fn with_option(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_options.insert(key.into(), value.into());
        self
    }
}

pub trait ModelSession: Send + Sync {
    fn device(&self) -> &Device;
    fn input_shapes(&self) -> &[TensorShape];
    fn output_shapes(&self) -> &[TensorShape];

    /// Execute inference using device-resident inputs, returning device-resident outputs
    fn run(&mut self, inputs: &[&dyn TensorBuffer]) -> Result<Vec<Box<dyn TensorBuffer>>, CoreError>;
}

pub trait Backend: Send + Sync {
    fn name(&self) -> &'static str;
    fn available_devices(&self) -> Vec<Device>;

    fn load_model(
        &self,
        model_bytes: &[u8],
        device: &Device,
    ) -> Result<Box<dyn ModelSession>, CoreError> {
        self.load_model_with_config(model_bytes, &SessionConfig::new(device.clone()))
    }

    fn load_model_from_file(
        &self,
        path: &str,
        device: &Device,
    ) -> Result<Box<dyn ModelSession>, CoreError> {
        let bytes = std::fs::read(path).map_err(|e| {
            CoreError::ModelLoadFailed(format!("Failed to read model file {}: {}", path, e))
        })?;
        self.load_model(&bytes, device)
    }

    fn load_model_with_config(
        &self,
        model_bytes: &[u8],
        config: &SessionConfig,
    ) -> Result<Box<dyn ModelSession>, CoreError>;
}
