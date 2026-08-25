use crate::error::ExecuTorchError;
pub use executorch::ndarray;
use executorch::tensor::ScalarType;
use infers_core::{
    AnyHostTensor, CoreError, CpuTensor, DataType, Device, TensorBuffer, TensorShape,
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
                std::mem::size_of_val(slice),
            )
        };
        Self::new(device, shape, DataType::F32, byte_slice.to_vec())
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.data.as_ref()
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
        match self.dtype {
            DataType::U8 => {
                let bytes = match Arc::try_unwrap(Arc::clone(&self.data)) {
                    Ok(vec) => vec,
                    Err(arc) => arc.as_ref().clone(),
                };
                Ok(Box::new(CpuTensor::from_u8(self.shape.clone(), bytes)?))
            }
            DataType::I32 => {
                let need = self.shape.byte_size(DataType::I32);
                let data = infers_core::bytes_to_vec::<i32>(&self.data.as_ref()[..need])?;
                Ok(Box::new(CpuTensor::from_i32(self.shape.clone(), data)?))
            }
            DataType::I64 => {
                let need = self.shape.byte_size(DataType::I64);
                let data = infers_core::bytes_to_vec::<i64>(&self.data.as_ref()[..need])?;
                Ok(Box::new(CpuTensor::from_i64(self.shape.clone(), data)?))
            }
            DataType::F32 => {
                let need = self.shape.byte_size(DataType::F32);
                let data = infers_core::bytes_to_vec::<f32>(&self.data.as_ref()[..need])?;
                Ok(Box::new(CpuTensor::from_f32(self.shape.clone(), data)?))
            }
            _ => {
                let bytes = match Arc::try_unwrap(Arc::clone(&self.data)) {
                    Ok(vec) => vec,
                    Err(arc) => arc.as_ref().clone(),
                };
                Ok(Box::new(CpuTensor::from_u8(
                    TensorShape::new(vec![bytes.len()])?,
                    bytes,
                )?))
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