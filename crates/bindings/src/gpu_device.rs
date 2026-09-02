use crate::device::DeviceInfo;
use crate::error::InfersError;
use crate::tensor::{CpuTensor, GpuTensor, DataType, TensorShape};
use crate::image::GpuImageProcessor;
use infers_gpu::{allocate_tensor, Vulkan};
use processing::GpuImageProcessor as CoreGpuImageProcessor;
use std::sync::Arc;

/// GPU execution device. Owns the shared Vulkan context for preprocessing and inference.
#[derive(uniffi::Object)]
pub struct GpuDevice {
    vulkan: Vulkan,
}

impl std::fmt::Debug for GpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuDevice")
            .field("info", self.vulkan.info())
            .finish()
    }
}

impl GpuDevice {
    pub(crate) fn vulkan(&self) -> &Vulkan {
        &self.vulkan
    }
}

#[uniffi::export]
impl GpuDevice {
    #[uniffi::constructor]
    pub fn new(id: u64) -> Result<Arc<Self>, InfersError> {
        #[cfg(target_os = "android")]
        {
            let info = infers_core::DeviceInfo {
                kind: infers_core::DeviceKind::Gpu,
                id: id as usize,
                name: format!("GPU:{}", id),
            };
            let context = Arc::new(
                platform_android::create_vulkan_context(&info).map_err(InfersError::from)?,
            );
            Ok(Arc::new(Self {
                vulkan: Vulkan::from_context(context),
            }))
        }
        #[cfg(not(target_os = "android"))]
        {
            let vulkan = Vulkan::new(id as usize).map_err(|err| InfersError::ProcessingFailed {
                reason: err.to_string(),
            })?;
            Ok(Arc::new(Self { vulkan }))
        }
    }

    pub fn info(&self) -> DeviceInfo {
        self.vulkan.info().clone().into()
    }

    pub fn upload_tensor(&self, tensor: Arc<CpuTensor>) -> Result<Arc<GpuTensor>, InfersError> {
        let gpu = tensor.inner().to_device(&self.vulkan).map_err(InfersError::from)?;
        Ok(Arc::new(GpuTensor::from_inner(gpu)))
    }
}

/// Create a GPU image processor bound to [device].
#[uniffi::export]
pub fn create_gpu_image_processor(
    device: Arc<GpuDevice>,
) -> Result<Arc<GpuImageProcessor>, InfersError> {
    let proc = CoreGpuImageProcessor::new(device.vulkan.clone()).map_err(InfersError::from)?;
    Ok(Arc::new(GpuImageProcessor { inner: proc }))
}

/// Allocate an uninitialized GPU tensor on [device].
#[uniffi::export]
pub fn create_gpu_tensor(
    device: Arc<GpuDevice>,
    shape: TensorShape,
    dtype: DataType,
) -> Result<Arc<GpuTensor>, InfersError> {
    let core_shape: infers_core::TensorShape = shape.try_into()?;
    let core_dtype: infers_core::DataType = dtype.into();
    let tensor = allocate_tensor(&device.vulkan, core_shape, core_dtype).map_err(InfersError::from)?;
    Ok(Arc::new(GpuTensor::from_inner(tensor)))
}
