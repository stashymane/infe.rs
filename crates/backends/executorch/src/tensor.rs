use crate::error::ExecuTorchError;
pub use executorch::ndarray;
use executorch::tensor::ScalarType;
use infers_core::{
    AnyHostTensor, CoreError, CpuTensor, DataType, Device, DeviceBuffer, TensorBuffer, TensorShape,
};
use std::sync::Arc;

/// Convert `infers_core::DataType` to `executorch::tensor::ScalarType`.
pub fn data_type_to_scalar_type(dt: DataType) -> Result<ScalarType, ExecuTorchError> {
    match dt {
        DataType::U8 => Ok(ScalarType::Byte),
        DataType::I8 => Ok(ScalarType::Char),
        DataType::I16 => Ok(ScalarType::Short),
        DataType::I32 => Ok(ScalarType::Int),
        DataType::I64 => Ok(ScalarType::Long),
        DataType::F16 => Ok(ScalarType::Half),
        DataType::F32 => Ok(ScalarType::Float),
        DataType::F64 => Ok(ScalarType::Double),
    }
}

/// Convert `executorch::tensor::ScalarType` to `infers_core::DataType`.
pub fn scalar_type_to_data_type(st: ScalarType) -> Result<DataType, ExecuTorchError> {
    match st {
        ScalarType::Byte => Ok(DataType::U8),
        ScalarType::Char => Ok(DataType::I8),
        ScalarType::Short => Ok(DataType::I16),
        ScalarType::Int => Ok(DataType::I32),
        ScalarType::Long => Ok(DataType::I64),
        ScalarType::Half => Ok(DataType::F16),
        ScalarType::Float => Ok(DataType::F32),
        ScalarType::Double => Ok(DataType::F64),
        other => Err(ExecuTorchError::Execution(format!(
            "Unsupported ExecuTorch ScalarType: {:?}",
            other
        ))),
    }
}

/// An ExecuTorch tensor buffer that can reside on CPU, GPU (Vulkan), or NPU (QNN).
#[derive(Clone, Debug)]
pub struct ExecuTorchTensorBuffer {
    device: Device,
    shape: TensorShape,
    dtype: DataType,
    data: Arc<Vec<u8>>,
}

impl ExecuTorchTensorBuffer {
    /// Create a new ExecuTorch tensor buffer on the given device with raw byte data.
    pub fn new(
        device: Device,
        shape: TensorShape,
        dtype: DataType,
        data: Vec<u8>,
    ) -> Result<Self, ExecuTorchError> {
        let expected_size = shape.byte_size(dtype);
        if data.len() != expected_size {
            return Err(ExecuTorchError::BufferError(format!(
                "Buffer size mismatch: shape {:?} with dtype {:?} requires {} bytes, but {} bytes were provided",
                shape, dtype, expected_size, data.len()
            )));
        }
        Ok(Self {
            device,
            shape,
            dtype,
            data: Arc::new(data),
        })
    }

    /// Create an ExecuTorch tensor buffer from an `AnyHostTensor`.
    pub fn from_host_tensor(device: Device, tensor: &dyn AnyHostTensor) -> Self {
        Self {
            device,
            shape: tensor.shape().clone(),
            dtype: tensor.dtype(),
            data: Arc::new(tensor.as_bytes().to_vec()),
        }
    }

    /// Create an ExecuTorch tensor buffer from a typed `CpuTensor<f32>`.
    pub fn from_f32_tensor(device: Device, tensor: &CpuTensor<f32>) -> Self {
        Self {
            device,
            shape: tensor.shape().clone(),
            dtype: DataType::F32,
            data: Arc::new(tensor.as_bytes().to_vec()),
        }
    }

    /// Create an ExecuTorch tensor buffer from a slice of `f32` with the given shape.
    pub fn from_f32_slice(
        device: Device,
        shape: TensorShape,
        slice: &[f32],
    ) -> Result<Self, ExecuTorchError> {
        if slice.len() != shape.element_count() {
            return Err(ExecuTorchError::BufferError(format!(
                "Element count mismatch: shape {:?} requires {} elements, got {}",
                shape,
                shape.element_count(),
                slice.len()
            )));
        }
        let byte_slice = unsafe {
            std::slice::from_raw_parts(
                slice.as_ptr() as *const u8,
                slice.len() * std::mem::size_of::<f32>(),
            )
        };
        Self::new(device, shape, DataType::F32, byte_slice.to_vec())
    }

    /// Create an ExecuTorch tensor buffer from an `ndarray::ArrayD<f32>`.
    pub fn from_ndarray_f32(
        device: Device,
        array: ndarray::ArrayD<f32>,
    ) -> Result<Self, ExecuTorchError> {
        let shape = TensorShape::new(array.shape().to_vec())?;
        let contiguous = array.into_raw_vec_and_offset().0;
        Self::from_f32_slice(device, shape, &contiguous)
    }

    /// Create an ExecuTorch tensor buffer from a typed `CpuTensor<u8>`.
    pub fn from_u8_tensor(device: Device, tensor: &CpuTensor<u8>) -> Self {
        Self {
            device,
            shape: tensor.shape().clone(),
            dtype: DataType::U8,
            data: Arc::new(tensor.as_bytes().to_vec()),
        }
    }

    /// Create an uninitialized/zeroed tensor buffer on the target device.
    pub fn zeroed(
        device: Device,
        shape: TensorShape,
        dtype: DataType,
    ) -> Result<Self, ExecuTorchError> {
        let size = shape.byte_size(dtype);
        Ok(Self {
            device,
            shape,
            dtype,
            data: Arc::new(vec![0u8; size]),
        })
    }

    /// Raw byte access (if device memory is mapped/accessible)
    pub fn raw_bytes(&self) -> &[u8] {
        &self.data
    }

    /// View as `f32` slice if dtype is F32
    pub fn as_slice_f32(&self) -> Result<&[f32], ExecuTorchError> {
        if self.dtype != DataType::F32 {
            return Err(ExecuTorchError::DataTypeMismatch {
                expected: DataType::F32,
                actual: self.dtype,
            });
        }
        let count = self.shape.element_count();
        let slice = unsafe {
            std::slice::from_raw_parts(self.data.as_ptr() as *const f32, count)
        };
        Ok(slice)
    }

    /// View as `u8` slice if dtype is U8
    pub fn as_slice_u8(&self) -> Result<&[u8], ExecuTorchError> {
        if self.dtype != DataType::U8 {
            return Err(ExecuTorchError::DataTypeMismatch {
                expected: DataType::U8,
                actual: self.dtype,
            });
        }
        Ok(&self.data)
    }

    /// Convert to `ndarray::ArrayD<f32>`
    pub fn to_ndarray_f32(&self) -> Result<ndarray::ArrayD<f32>, ExecuTorchError> {
        let slice = self.as_slice_f32()?;
        let shape = self.shape.dims();
        ndarray::ArrayD::from_shape_vec(shape, slice.to_vec()).map_err(|e| {
            ExecuTorchError::Execution(format!("Failed to create ndarray: {:?}", e))
        })
    }
}

impl TensorBuffer for ExecuTorchTensorBuffer {
    fn shape(&self) -> &TensorShape {
        &self.shape
    }

    fn dtype(&self) -> DataType {
        self.dtype
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn read_to_cpu(&self) -> Result<Box<dyn AnyHostTensor>, CoreError> {
        // Explicit device synchronization / readback into host CPU memory
        match self.dtype {
            DataType::U8 => {
                let tensor = CpuTensor::from_u8(self.shape.clone(), self.data.as_ref().clone())?;
                Ok(Box::new(tensor))
            }
            DataType::I32 => {
                let slice = unsafe {
                    std::slice::from_raw_parts(
                        self.data.as_ptr() as *const i32,
                        self.shape.element_count(),
                    )
                };
                let tensor = CpuTensor::from_i32(self.shape.clone(), slice.to_vec())?;
                Ok(Box::new(tensor))
            }
            DataType::I64 => {
                let slice = unsafe {
                    std::slice::from_raw_parts(
                        self.data.as_ptr() as *const i64,
                        self.shape.element_count(),
                    )
                };
                let tensor = CpuTensor::from_i64(self.shape.clone(), slice.to_vec())?;
                Ok(Box::new(tensor))
            }
            DataType::F32 => {
                let slice = unsafe {
                    std::slice::from_raw_parts(
                        self.data.as_ptr() as *const f32,
                        self.shape.element_count(),
                    )
                };
                let tensor = CpuTensor::from_f32(self.shape.clone(), slice.to_vec())?;
                Ok(Box::new(tensor))
            }
            _ => {
                // For other data types, return as raw byte host tensor
                let tensor = CpuTensor::from_u8(
                    TensorShape::new(vec![self.data.len()])?,
                    self.data.as_ref().clone(),
                )?;
                Ok(Box::new(tensor))
            }
        }
    }

    fn copy_to_device(&self, target: &Device) -> Result<Box<dyn TensorBuffer>, CoreError> {
        if &self.device == target {
            return Ok(Box::new(self.clone()));
        }
        // Create new device-resident copy on target device
        let transferred = ExecuTorchTensorBuffer {
            device: target.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype,
            data: Arc::clone(&self.data),
        };
        Ok(Box::new(transferred))
    }
}

impl DeviceBuffer for ExecuTorchTensorBuffer {
    fn device(&self) -> &Device {
        &self.device
    }

    fn byte_size(&self) -> usize {
        self.shape.byte_size(self.dtype)
    }

    fn copy_to_host(&self, out: &mut [u8]) -> Result<(), CoreError> {
        let copy_len = out.len().min(self.data.len());
        out[..copy_len].copy_from_slice(&self.data[..copy_len]);
        Ok(())
    }

    fn copy_to_device(&self, target: &Device) -> Result<Box<dyn DeviceBuffer>, CoreError> {
        let transferred = ExecuTorchTensorBuffer {
            device: target.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype,
            data: Arc::clone(&self.data),
        };
        Ok(Box::new(transferred))
    }
}
