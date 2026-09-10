use crate::error::InfersError;
use crate::session::CpuSession;
use infers_backend_executorch::{
    config::XnnpackOptions as CoreXnnpackOptions,
    ExecuTorchBackend as CoreExecuTorchBackend,
};
use std::sync::Arc;

#[cfg(feature = "vulkan")]
use crate::gpu_device::GpuDevice;
#[cfg(feature = "vulkan")]
use crate::session::GpuSession;
#[cfg(feature = "vulkan")]
use infers_backend_executorch::config::VulkanOptions as CoreVulkanOptions;

#[derive(Debug, Clone, uniffi::Record)]
pub struct XnnpackOptions {
    pub num_threads: u32,
    pub method: Option<String>,
}

impl From<XnnpackOptions> for CoreXnnpackOptions {
    fn from(opts: XnnpackOptions) -> Self {
        Self {
            num_threads: opts.num_threads.max(1) as usize,
            method: opts.method,
        }
    }
}

#[cfg(feature = "vulkan")]
#[derive(Debug, Clone, uniffi::Record)]
pub struct VulkanOptions {
    pub method: Option<String>,
}

#[cfg(feature = "vulkan")]
impl From<VulkanOptions> for CoreVulkanOptions {
    fn from(opts: VulkanOptions) -> Self {
        Self {
            method: opts.method,
        }
    }
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

    pub fn load_xnnpack(
        &self,
        model_bytes: Vec<u8>,
        opts: XnnpackOptions,
    ) -> Result<Arc<CpuSession>, InfersError> {
        let session = self
            .inner
            .load_xnnpack(&model_bytes, opts.into())
            .map_err(InfersError::from)?;
        Ok(Arc::new(CpuSession::new(session)))
    }

    pub fn load_xnnpack_from_file(
        &self,
        path: String,
        opts: XnnpackOptions,
    ) -> Result<Arc<CpuSession>, InfersError> {
        let session = self
            .inner
            .load_xnnpack_from_file(&path, opts.into())
            .map_err(InfersError::from)?;
        Ok(Arc::new(CpuSession::new(session)))
    }
}

#[cfg(feature = "vulkan")]
#[uniffi::export]
impl FfiBackend {
    pub fn load_vulkan(
        &self,
        model_bytes: Vec<u8>,
        device: Arc<GpuDevice>,
        opts: VulkanOptions,
    ) -> Result<Arc<GpuSession>, InfersError> {
        let session = self
            .inner
            .load_vulkan(&model_bytes, device.vulkan(), opts.into())
            .map_err(InfersError::from)?;
        Ok(Arc::new(GpuSession::new(session)))
    }

    pub fn load_vulkan_from_file(
        &self,
        path: String,
        device: Arc<GpuDevice>,
        opts: VulkanOptions,
    ) -> Result<Arc<GpuSession>, InfersError> {
        let session = self
            .inner
            .load_vulkan_from_file(&path, device.vulkan(), opts.into())
            .map_err(InfersError::from)?;
        Ok(Arc::new(GpuSession::new(session)))
    }
}
