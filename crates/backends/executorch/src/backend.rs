use crate::config::ExecuTorchBackendConfig;
use crate::session::ExecuTorchSession;
use infers_core::{Backend, CoreError, Device, ModelSession, SessionConfig};

/// The ExecuTorch inference backend.
#[derive(Clone, Debug, Default)]
pub struct ExecuTorchBackend;

impl ExecuTorchBackend {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Load a `.pte` program with an explicit backend configuration.
    ///
    /// For [`ExecuTorchBackendConfig::Vulkan`], pass an existing
    /// [`infers_gpu::VulkanContext`] created by the app (required for teardown order
    /// and sharing with GPU preprocessing).
    pub fn load_model(
        &self,
        model_bytes: &[u8],
        config: ExecuTorchBackendConfig,
    ) -> Result<Box<dyn ModelSession>, CoreError> {
        let session = ExecuTorchSession::load(model_bytes, &config).map_err(CoreError::from)?;
        Ok(Box::new(session))
    }

    /// Load a `.pte` program from a file path with an explicit backend configuration.
    ///
    /// The file at `path` must remain readable for the lifetime of the returned session.
    pub fn load_model_from_file(
        &self,
        path: &str,
        config: ExecuTorchBackendConfig,
    ) -> Result<Box<dyn ModelSession>, CoreError> {
        let session =
            ExecuTorchSession::load_from_path(std::path::Path::new(path), &config)
                .map_err(CoreError::from)?;
        Ok(Box::new(session))
    }
}

impl Backend for ExecuTorchBackend {
    fn name(&self) -> &'static str {
        "ExecuTorch"
    }

    fn available_devices(&self) -> Vec<Device> {
        let mut devices = vec![Device::cpu()];
        #[cfg(feature = "vulkan")]
        {
            devices.push(Device::gpu(0));
            devices.push(Device::npu(0));
        }
        devices
    }

    fn load_model_with_config(
        &self,
        model_bytes: &[u8],
        config: &SessionConfig,
    ) -> Result<Box<dyn ModelSession>, CoreError> {
        // Core trait path: CPU/XNNPACK only. GPU requires [`ExecuTorchBackendConfig::Vulkan`].
        if config.device.is_gpu() {
            return Err(CoreError::ModelLoadFailed(
                "GPU ExecuTorch models require ExecuTorchBackend::load_model with \
                 ExecuTorchBackendConfig::Vulkan { context, .. }"
                    .into(),
            ));
        }
        let method = config.extra_options.get("method").cloned();
        let num_threads = config.num_threads.max(1);
        self.load_model(
            model_bytes,
            ExecuTorchBackendConfig::Xnnpack {
                num_threads,
                method,
            },
        )
    }
}
