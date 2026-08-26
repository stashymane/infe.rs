use crate::device::Device;
use crate::error::InfersError;
#[cfg(feature = "vulkan")]
use crate::gpu_context::GpuContext;
use crate::session::ModelSession;
use infers_backend_executorch::{
    ExecuTorchBackend as CoreExecuTorchBackend, ExecuTorchBackendConfig,
};
use infers_core::Backend;
use std::sync::Arc;

#[derive(Debug, Clone, uniffi::Enum)]
pub enum BackendConfig {
    #[cfg(feature = "xnnpack")]
    Xnnpack {
        num_threads: u32,
        method: Option<String>,
    },
    #[cfg(feature = "vulkan")]
    Vulkan {
        context: Arc<GpuContext>,
        method: Option<String>,
    },
}

#[derive(uniffi::Object)]
pub struct FfiBackend {
    inner: CoreExecuTorchBackend,
}

impl std::fmt::Debug for FfiBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FfiBackend").finish()
    }
}

#[uniffi::export]
impl FfiBackend {
    #[uniffi::constructor]
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: CoreExecuTorchBackend::new(),
        })
    }

    pub fn available_devices(&self) -> Vec<Device> {
        self.inner
            .available_devices()
            .into_iter()
            .map(Into::into)
            .collect()
    }

    pub fn load_model(
        &self,
        model_bytes: Vec<u8>,
        config: BackendConfig,
    ) -> Result<Arc<ModelSession>, InfersError> {
        let core_config = to_core_config(config)?;
        let session = self
            .inner
            .load_model(&model_bytes, core_config)
            .map_err(InfersError::from)?;
        Ok(Arc::new(ModelSession::new(session)))
    }

    pub fn load_model_from_file(
        &self,
        path: String,
        config: BackendConfig,
    ) -> Result<Arc<ModelSession>, InfersError> {
        let core_config = to_core_config(config)?;
        let session = self
            .inner
            .load_model_from_file(&path, core_config)
            .map_err(InfersError::from)?;
        Ok(Arc::new(ModelSession::new(session)))
    }
}

fn to_core_config(config: BackendConfig) -> Result<ExecuTorchBackendConfig, InfersError> {
    Ok(match config {
        #[cfg(feature = "xnnpack")]
        BackendConfig::Xnnpack {
            num_threads,
            method,
        } => ExecuTorchBackendConfig::Xnnpack {
            num_threads: num_threads.max(1) as usize,
            method,
        },
        #[cfg(feature = "vulkan")]
        BackendConfig::Vulkan { context, method } => ExecuTorchBackendConfig::Vulkan {
            context: Arc::clone(context.inner()),
            method,
        },
    })
}
