use crate::device::Device;
use crate::error::InfersError;
use crate::image::ImageProcessor;
use infers_gpu::VulkanContext;
use processing::GpuImageProcessor;
use std::sync::Arc;

/// Opaque shared Vulkan context. Create before GPU processors or Vulkan inference,
/// and drop sessions/processors before the last context handle.
#[derive(uniffi::Object)]
pub struct GpuContext {
    inner: Arc<VulkanContext>,
}

impl std::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContext")
            .field("device", self.inner.logical_device())
            .finish()
    }
}

#[uniffi::export]
impl GpuContext {
    pub fn device(&self) -> Device {
        self.inner.logical_device().clone().into()
    }
}

impl GpuContext {
    pub(crate) fn inner(&self) -> &Arc<VulkanContext> {
        &self.inner
    }
}

/// Create a Vulkan context for GPU preprocessing and/or ExecuTorch Vulkan inference.
#[uniffi::export]
pub fn create_gpu_context(device: Device) -> Result<Arc<GpuContext>, InfersError> {
    let core_device: infers_core::Device = device.into();
    #[cfg(target_os = "android")]
    let context = Arc::new(
        platform_android::create_vulkan_context(&core_device).map_err(InfersError::from)?,
    );
    #[cfg(not(target_os = "android"))]
    let context = Arc::new(
        VulkanContext::new(&core_device).map_err(|err| InfersError::ProcessingFailed {
            message: err.to_string(),
        })?,
    );
    Ok(Arc::new(GpuContext { inner: context }))
}

/// Create a GPU image processor that shares `context`.
#[uniffi::export]
pub fn create_gpu_image_processor(context: Arc<GpuContext>) -> Result<Arc<ImageProcessor>, InfersError> {
    let device: Device = context.inner.logical_device().clone().into();
    let gpu_proc = GpuImageProcessor::new(Arc::clone(context.inner())).map_err(InfersError::from)?;
    Ok(Arc::new(ImageProcessor {
        cpu_proc: None,
        gpu_proc: Some(gpu_proc),
        device,
    }))
}
