use crate::delegate::ExecuTorchDelegate;
use crate::program::{MethodDescriptor, ProgramMetadata};
use crate::session::ExecuTorchSession;
use infers_core::{Backend, CoreError, Device, ModelSession, SessionConfig};

/// The ExecuTorch inference backend.
#[derive(Clone, Debug, Default)]
pub struct ExecuTorchBackend;

impl ExecuTorchBackend {
    pub fn new() -> Self {
        Self
    }

    /// Load directly from an existing `MethodDescriptor` with session config
    pub fn load_method(
        &self,
        method: MethodDescriptor,
        config: &SessionConfig,
    ) -> Result<Box<dyn ModelSession>, CoreError> {
        let delegate = if let Some(del_name) = config.extra_options.get("delegate") {
            match del_name.to_lowercase().as_str() {
                "xnnpack" => ExecuTorchDelegate::Xnnpack,
                "vulkan" => ExecuTorchDelegate::Vulkan,
                "qnn" => ExecuTorchDelegate::Qnn,
                "coreml" => ExecuTorchDelegate::CoreMl,
                "portable_cpu" => ExecuTorchDelegate::PortableCpu,
                other => ExecuTorchDelegate::Custom(other.to_string()),
            }
        } else {
            ExecuTorchDelegate::for_device(&config.device)
        };

        let session = ExecuTorchSession::new(config.device.clone(), delegate, method)
            .map_err(CoreError::from)?;

        Ok(Box::new(session))
    }

    /// Load directly from `ProgramMetadata` with session config
    pub fn load_metadata_with_config(
        &self,
        metadata: ProgramMetadata,
        config: &SessionConfig,
    ) -> Result<Box<dyn ModelSession>, CoreError> {
        let method_name = config
            .extra_options
            .get("method")
            .map(|s| s.as_str())
            .unwrap_or("forward");

        let method = metadata
            .method(method_name)
            .cloned()
            .ok_or_else(|| {
                CoreError::ModelLoadFailed(format!(
                    "Method '{}' not found in ExecuTorch program metadata",
                    method_name
                ))
            })?;

        self.load_method(method, config)
    }

    /// Load a model from a file path with session configuration
    pub fn load_model_from_file_with_config(
        &self,
        path: &str,
        config: &SessionConfig,
    ) -> Result<Box<dyn ModelSession>, CoreError> {
        let bytes = std::fs::read(path).map_err(|e| {
            CoreError::ModelLoadFailed(format!("Failed to read model file '{}': {}", path, e))
        })?;
        self.load_model_with_config(&bytes, config)
    }
}

impl Backend for ExecuTorchBackend {
    fn name(&self) -> &'static str {
        "ExecuTorch"
    }

    fn available_devices(&self) -> Vec<Device> {
        vec![Device::cpu(), Device::gpu(0), Device::npu(0)]
    }

    fn load_model_with_config(
        &self,
        model_bytes: &[u8],
        config: &SessionConfig,
    ) -> Result<Box<dyn ModelSession>, CoreError> {
        let metadata = ProgramMetadata::from_bytes(model_bytes)
            .map_err(|e| CoreError::ModelLoadFailed(e.to_string()))?;
        self.load_metadata_with_config(metadata, config)
    }
}
