use crate::device::Device;
use crate::error::InfersError;
use crate::gpu_context::GpuContext;
use infers_core::{
    bytes_to_vec, CpuTensor, DataType as CoreDataType, DeviceTransfer, TensorBuffer as CoreTensorBuffer,
    TensorShape as CoreTensorShape,
};
use processing::VulkanDeviceTransfer;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, uniffi::Enum)]
pub enum DataType {
    U8,
    I8,
    I16,
    I32,
    I64,
    F16,
    F32,
    F64,
}

uniffi_mirror! {
    DataType <=> CoreDataType,
    [U8, I8, I16, I32, I64, F16, F32, F64]
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, uniffi::Record)]
pub struct TensorShape {
    pub dims: Vec<u64>,
}

impl From<CoreTensorShape> for TensorShape {
    fn from(shape: CoreTensorShape) -> Self {
        Self {
            dims: shape.dims().iter().map(|&d| d as u64).collect(),
        }
    }
}

impl TryFrom<TensorShape> for CoreTensorShape {
    type Error = InfersError;

    fn try_from(shape: TensorShape) -> Result<Self, Self::Error> {
        let dims: Vec<usize> = shape.dims.into_iter().map(|d| d as usize).collect();
        CoreTensorShape::new(dims).map_err(InfersError::from)
    }
}

#[derive(Debug, uniffi::Object)]
pub struct TensorBuffer {
    inner: Arc<dyn CoreTensorBuffer>,
}

impl TensorBuffer {
    pub fn from_boxed(boxed: Box<dyn CoreTensorBuffer>) -> Self {
        Self {
            inner: Arc::from(boxed),
        }
    }

    pub fn as_core(&self) -> &dyn CoreTensorBuffer {
        self.inner.as_ref()
    }
}

#[uniffi::export]
impl TensorBuffer {
    pub fn shape(&self) -> TensorShape {
        self.inner.shape().clone().into()
    }

    pub fn dtype(&self) -> DataType {
        self.inner.dtype().into()
    }

    pub fn device(&self) -> Device {
        self.inner.device().clone().into()
    }

    pub fn byte_size(&self) -> u64 {
        self.inner.byte_size() as u64
    }

    /// Raw little-endian bytes of the tensor contents after CPU readback.
    pub fn read_to_cpu_bytes(&self) -> Result<Vec<u8>, InfersError> {
        let host = self.inner.read_to_cpu().map_err(InfersError::from)?;
        Ok(host.as_bytes().to_vec())
    }

    pub fn copy_to_device(
        &self,
        target: Device,
        gpu_context: Option<Arc<GpuContext>>,
    ) -> Result<Arc<TensorBuffer>, InfersError> {
        let target_device: infers_core::Device = target.into();
        let transfer: Option<VulkanDeviceTransfer> = gpu_context
            .as_ref()
            .map(|ctx| VulkanDeviceTransfer::new(Arc::clone(ctx.inner())));
        let transferred = self.inner.copy_to_device(
            &target_device,
            transfer.as_ref().map(|t| t as &dyn DeviceTransfer),
        )?;
        Ok(Arc::new(TensorBuffer::from_boxed(transferred)))
    }
}

/// Upload a host-resident tensor to GPU memory using `gpu_context`.
#[uniffi::export]
pub fn upload_to_gpu(
    gpu_context: Arc<GpuContext>,
    tensor: Arc<TensorBuffer>,
) -> Result<Arc<TensorBuffer>, InfersError> {
    let transfer = VulkanDeviceTransfer::new(Arc::clone(gpu_context.inner()));
    let device = gpu_context.device();
    let target: infers_core::Device = device.into();
    let uploaded = tensor
        .inner
        .copy_to_device(&target, Some(&transfer))?;
    Ok(Arc::new(TensorBuffer::from_boxed(uploaded)))
}

/// Construct a CPU tensor from little-endian raw bytes matching [dtype].
#[uniffi::export]
pub fn create_tensor_from_bytes(
    shape: TensorShape,
    dtype: DataType,
    data: Vec<u8>,
) -> Result<Arc<TensorBuffer>, InfersError> {
    let core_shape: CoreTensorShape = shape.try_into()?;
    let core_dtype: CoreDataType = dtype.into();
    let tensor: Box<dyn CoreTensorBuffer> = match core_dtype {
        CoreDataType::U8 => {
            Box::new(CpuTensor::new(core_shape, core_dtype, data).map_err(InfersError::from)?)
        }
        CoreDataType::F32 => {
            let values = bytes_to_vec::<f32>(&data).map_err(InfersError::from)?;
            Box::new(CpuTensor::new(core_shape, core_dtype, values).map_err(InfersError::from)?)
        }
        CoreDataType::I32 => {
            let values = bytes_to_vec::<i32>(&data).map_err(InfersError::from)?;
            Box::new(CpuTensor::new(core_shape, core_dtype, values).map_err(InfersError::from)?)
        }
        CoreDataType::I64 => {
            let values = bytes_to_vec::<i64>(&data).map_err(InfersError::from)?;
            Box::new(CpuTensor::new(core_shape, core_dtype, values).map_err(InfersError::from)?)
        }
        other => {
            return Err(InfersError::UnsupportedType {
                reason: format!("create_tensor_from_bytes does not support {other:?}"),
            });
        }
    };
    Ok(Arc::new(TensorBuffer::from_boxed(tensor)))
}
