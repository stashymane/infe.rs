use crate::device::DeviceInfo;
use crate::error::InfersError;
use infers_core::{
    bytes_to_vec, Cpu, DataType as CoreDataType, HostTensor, Tensor,
    TensorShape as CoreTensorShape,
};
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

/// Host-resident CPU tensor.
#[derive(uniffi::Object)]
pub struct CpuTensor {
    inner: Tensor<Cpu>,
}

impl CpuTensor {
    pub(crate) fn inner(&self) -> &Tensor<Cpu> {
        &self.inner
    }

    pub(crate) fn from_inner(inner: Tensor<Cpu>) -> Self {
        Self { inner }
    }
}

#[uniffi::export]
impl CpuTensor {
    pub fn shape(&self) -> TensorShape {
        self.inner.shape().clone().into()
    }

    pub fn dtype(&self) -> DataType {
        self.inner.dtype().into()
    }

    pub fn device_info(&self) -> DeviceInfo {
        infers_core::Cpu::info().clone().into()
    }

    pub fn byte_size(&self) -> u64 {
        self.inner.byte_size() as u64
    }

    pub fn read_bytes(&self) -> Result<Vec<u8>, InfersError> {
        self.copy_bytes(0, self.inner.byte_size() as u64)
    }

    /// Copy a byte range `[byte_offset, byte_offset + len)`.
    pub fn copy_bytes(&self, byte_offset: u64, len: u64) -> Result<Vec<u8>, InfersError> {
        let host = self.inner.read_to_host().map_err(InfersError::from)?;
        host.copy_bytes_range(byte_offset as usize, len as usize)
            .map_err(InfersError::from)
    }

    /// Copy `len` f32 elements starting at `start_elem`.
    pub fn copy_f32(&self, start_elem: u64, len: u64) -> Result<Vec<f32>, InfersError> {
        let host = self.inner.read_to_host().map_err(InfersError::from)?;
        host.copy_f32_range(start_elem as usize, len as usize)
            .map_err(InfersError::from)
    }

    /// Copy `len` i32 elements starting at `start_elem`.
    pub fn copy_i32(&self, start_elem: u64, len: u64) -> Result<Vec<i32>, InfersError> {
        let host = self.inner.read_to_host().map_err(InfersError::from)?;
        host.copy_i32_range(start_elem as usize, len as usize)
            .map_err(InfersError::from)
    }

    /// Copy `len` i64 elements starting at `start_elem`.
    pub fn copy_i64(&self, start_elem: u64, len: u64) -> Result<Vec<i64>, InfersError> {
        let host = self.inner.read_to_host().map_err(InfersError::from)?;
        host.copy_i64_range(start_elem as usize, len as usize)
            .map_err(InfersError::from)
    }

    /// Host data pointer as `u64` for JNI DirectByteBuffer mapping.
    ///
    /// Valid only while this [`CpuTensor`] handle is alive. Returns `(ptr, nbytes)`.
    pub fn host_data_ptr(&self) -> Result<HostDataPtr, InfersError> {
        let (ptr, len) = self.inner.host_ptr_len();
        Ok(HostDataPtr {
            ptr: ptr as u64,
            nbytes: len as u64,
        })
    }
}

/// Opaque host buffer address for DirectByteBuffer bridging.
#[derive(Debug, Clone, PartialEq, Eq, uniffi::Record)]
pub struct HostDataPtr {
    pub ptr: u64,
    pub nbytes: u64,
}

/// Construct a CPU tensor from little-endian raw bytes matching [dtype].
#[uniffi::export]
pub fn create_cpu_tensor_from_bytes(
    shape: TensorShape,
    dtype: DataType,
    data: Vec<u8>,
) -> Result<Arc<CpuTensor>, InfersError> {
    let core_shape: CoreTensorShape = shape.try_into()?;
    let core_dtype: CoreDataType = dtype.into();
    let host = match core_dtype {
        CoreDataType::U8 => HostTensor::from_u8(core_shape, data).map_err(InfersError::from)?,
        CoreDataType::F32 => {
            let values = bytes_to_vec::<f32>(&data).map_err(InfersError::from)?;
            HostTensor::from_f32(core_shape, values).map_err(InfersError::from)?
        }
        CoreDataType::I32 => {
            let values = bytes_to_vec::<i32>(&data).map_err(InfersError::from)?;
            HostTensor::from_i32(core_shape, values).map_err(InfersError::from)?
        }
        CoreDataType::I64 => {
            let values = bytes_to_vec::<i64>(&data).map_err(InfersError::from)?;
            HostTensor::from_i64(core_shape, values).map_err(InfersError::from)?
        }
        other => {
            return Err(InfersError::UnsupportedType {
                reason: format!("create_cpu_tensor_from_bytes does not support {other:?}"),
            });
        }
    };
    let tensor = Tensor::adopt_host(&host);
    Ok(Arc::new(CpuTensor::from_inner(tensor)))
}

#[cfg(feature = "vulkan")]
use infers_gpu::Vulkan;

#[cfg(feature = "vulkan")]
/// GPU-resident tensor.
#[derive(uniffi::Object)]
pub struct GpuTensor {
    inner: Tensor<Vulkan>,
}

#[cfg(feature = "vulkan")]
impl GpuTensor {
    pub(crate) fn inner(&self) -> &Tensor<Vulkan> {
        &self.inner
    }

    pub(crate) fn from_inner(inner: Tensor<Vulkan>) -> Self {
        Self { inner }
    }
}

#[cfg(feature = "vulkan")]
#[uniffi::export]
impl GpuTensor {
    pub fn shape(&self) -> TensorShape {
        self.inner.shape().clone().into()
    }

    pub fn dtype(&self) -> DataType {
        self.inner.dtype().into()
    }

    pub fn device_info(&self) -> DeviceInfo {
        self.inner.device().info().clone().into()
    }

    pub fn byte_size(&self) -> u64 {
        self.inner.byte_size() as u64
    }

    pub fn read_bytes(&self) -> Result<Vec<u8>, InfersError> {
        self.copy_bytes(0, self.inner.byte_size() as u64)
    }

    pub fn copy_bytes(&self, byte_offset: u64, len: u64) -> Result<Vec<u8>, InfersError> {
        let host = self.inner.read_to_host().map_err(InfersError::from)?;
        host.copy_bytes_range(byte_offset as usize, len as usize)
            .map_err(InfersError::from)
    }

    pub fn copy_f32(&self, start_elem: u64, len: u64) -> Result<Vec<f32>, InfersError> {
        let host = self.inner.read_to_host().map_err(InfersError::from)?;
        host.copy_f32_range(start_elem as usize, len as usize)
            .map_err(InfersError::from)
    }

    pub fn copy_i32(&self, start_elem: u64, len: u64) -> Result<Vec<i32>, InfersError> {
        let host = self.inner.read_to_host().map_err(InfersError::from)?;
        host.copy_i32_range(start_elem as usize, len as usize)
            .map_err(InfersError::from)
    }

    pub fn copy_i64(&self, start_elem: u64, len: u64) -> Result<Vec<i64>, InfersError> {
        let host = self.inner.read_to_host().map_err(InfersError::from)?;
        host.copy_i64_range(start_elem as usize, len as usize)
            .map_err(InfersError::from)
    }

    pub fn download(&self) -> Result<Arc<CpuTensor>, InfersError> {
        let cpu = self
            .inner
            .to_device(&Cpu)
            .map_err(InfersError::from)?;
        Ok(Arc::new(CpuTensor::from_inner(cpu)))
    }
}
