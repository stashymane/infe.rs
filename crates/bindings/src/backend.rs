use crate::device::Device;
use crate::error::InfersError;
use crate::session::ModelSession;
use infers_backend_executorch::ExecuTorchBackend as CoreExecuTorchBackend;
use infers_core::{Backend, MockBackend as CoreMockBackend};
use std::sync::Arc;

#[derive(uniffi::Object)]
pub struct ExecuTorchBackend {
    inner: CoreExecuTorchBackend,
}

impl std::fmt::Debug for ExecuTorchBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecuTorchBackend").finish()
    }
}

#[uniffi::export]
impl ExecuTorchBackend {
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
        device: Device,
    ) -> Result<Arc<ModelSession>, InfersError> {
        let core_device: infers_core::Device = device.into();
        let session = self
            .inner
            .load_model(&model_bytes, &core_device)
            .map_err(InfersError::from)?;
        Ok(Arc::new(ModelSession::new(session)))
    }

    pub fn load_model_from_file(
        &self,
        path: String,
        device: Device,
    ) -> Result<Arc<ModelSession>, InfersError> {
        let core_device: infers_core::Device = device.into();
        let session = self
            .inner
            .load_model_from_file(&path, &core_device)
            .map_err(InfersError::from)?;
        Ok(Arc::new(ModelSession::new(session)))
    }
}

#[derive(uniffi::Object)]
pub struct MockBackend {
    inner: CoreMockBackend,
}

impl std::fmt::Debug for MockBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockBackend").finish()
    }
}

#[uniffi::export]
impl MockBackend {
    #[uniffi::constructor]
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: CoreMockBackend::new("mock_backend"),
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
        device: Device,
    ) -> Result<Arc<ModelSession>, InfersError> {
        let core_device: infers_core::Device = device.into();
        let session = self
            .inner
            .load_model(&model_bytes, &core_device)
            .map_err(InfersError::from)?;
        Ok(Arc::new(ModelSession::new(session)))
    }
}
