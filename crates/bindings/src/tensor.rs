use crate::device::Device;
use crate::error::InfersError;
use infers_core::{CpuTensor, DataType as CoreDataType, TensorBuffer as CoreTensorBuffer, TensorShape as CoreTensorShape};
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

    pub fn read_to_cpu_u8(&self) -> Result<Vec<u8>, InfersError> {
        let host = self.inner.read_to_cpu().map_err(InfersError::from)?;
        let slice = host.as_slice_u8().map_err(InfersError::from)?;
        Ok(slice.to_vec())
    }

    pub fn read_to_cpu_f32(&self) -> Result<Vec<f32>, InfersError> {
        let host = self.inner.read_to_cpu().map_err(InfersError::from)?;
        let slice = host.as_slice_f32().map_err(InfersError::from)?;
        Ok(slice.to_vec())
    }

    pub fn read_to_cpu_i32(&self) -> Result<Vec<i32>, InfersError> {
        let host = self.inner.read_to_cpu().map_err(InfersError::from)?;
        let slice = host.as_slice_i32().map_err(InfersError::from)?;
        Ok(slice.to_vec())
    }

    pub fn read_to_cpu_i64(&self) -> Result<Vec<i64>, InfersError> {
        let host = self.inner.read_to_cpu().map_err(InfersError::from)?;
        let slice = host.as_slice_i64().map_err(InfersError::from)?;
        Ok(slice.to_vec())
    }

    pub fn copy_to_device(&self, target: Device) -> Result<Arc<TensorBuffer>, InfersError> {
        let target_device: infers_core::Device = target.into();
        let transferred = self
            .inner
            .copy_to_device(&target_device)
            .map_err(InfersError::from)?;
        Ok(Arc::new(TensorBuffer::from_boxed(transferred)))
    }
}

#[uniffi::export]
pub fn create_tensor_from_f32(
    shape: TensorShape,
    data: Vec<f32>,
) -> Result<Arc<TensorBuffer>, InfersError> {
    let core_shape: CoreTensorShape = shape.try_into()?;
    let tensor = CpuTensor::new(core_shape, CoreDataType::F32, data).map_err(InfersError::from)?;
    Ok(Arc::new(TensorBuffer::from_boxed(Box::new(tensor))))
}

#[uniffi::export]
pub fn create_tensor_from_u8(
    shape: TensorShape,
    data: Vec<u8>,
) -> Result<Arc<TensorBuffer>, InfersError> {
    let core_shape: CoreTensorShape = shape.try_into()?;
    let tensor = CpuTensor::new(core_shape, CoreDataType::U8, data).map_err(InfersError::from)?;
    Ok(Arc::new(TensorBuffer::from_boxed(Box::new(tensor))))
}

#[uniffi::export]
pub fn create_tensor_from_i32(
    shape: TensorShape,
    data: Vec<i32>,
) -> Result<Arc<TensorBuffer>, InfersError> {
    let core_shape: CoreTensorShape = shape.try_into()?;
    let tensor = CpuTensor::new(core_shape, CoreDataType::I32, data).map_err(InfersError::from)?;
    Ok(Arc::new(TensorBuffer::from_boxed(Box::new(tensor))))
}

#[uniffi::export]
pub fn create_tensor_from_i64(
    shape: TensorShape,
    data: Vec<i64>,
) -> Result<Arc<TensorBuffer>, InfersError> {
    let core_shape: CoreTensorShape = shape.try_into()?;
    let tensor = CpuTensor::new(core_shape, CoreDataType::I64, data).map_err(InfersError::from)?;
    Ok(Arc::new(TensorBuffer::from_boxed(Box::new(tensor))))
}
