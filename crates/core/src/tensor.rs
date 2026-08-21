use std::any::Any;
use crate::device::Device;
use crate::error::CoreError;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

impl DataType {
    #[inline]
    pub const fn element_size(&self) -> usize {
        match self {
            Self::U8 | Self::I8 => 1,
            Self::I16 | Self::F16 => 2,
            Self::I32 | Self::F32 => 4,
            Self::I64 | Self::F64 => 8,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorShape {
    dims: Vec<usize>,
}

impl TensorShape {
    pub fn new(dims: impl Into<Vec<usize>>) -> Result<Self, CoreError> {
        let dims = dims.into();
        if dims.is_empty() {
            return Err(CoreError::InvalidShape("Dimensions cannot be empty".to_string()));
        }
        for (i, &d) in dims.iter().enumerate() {
            if d == 0 {
                return Err(CoreError::InvalidShape(format!(
                    "Dimension at index {} cannot be 0",
                    i
                )));
            }
        }
        Ok(Self { dims })
    }

    #[inline]
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    #[inline]
    pub fn element_count(&self) -> usize {
        self.dims.iter().copied().product()
    }

    #[inline]
    pub fn byte_size(&self, dtype: DataType) -> usize {
        self.element_count() * dtype.element_size()
    }
}

impl From<Vec<usize>> for TensorShape {
    fn from(dims: Vec<usize>) -> Self {
        Self::new(dims).expect("Valid tensor shape")
    }
}

impl<const N: usize> From<[usize; N]> for TensorShape {
    fn from(dims: [usize; N]) -> Self {
        Self::new(dims.to_vec()).expect("Valid tensor shape")
    }
}

/// A trait for inspecting and extracting host CPU tensor data
pub trait AnyHostTensor: Send + Sync + std::fmt::Debug {
    fn shape(&self) -> &TensorShape;
    fn dtype(&self) -> DataType;
    fn as_bytes(&self) -> &[u8];
    fn as_slice_f32(&self) -> Result<&[f32], CoreError>;
    fn as_slice_u8(&self) -> Result<&[u8], CoreError>;
    fn as_slice_i32(&self) -> Result<&[i32], CoreError>;
    fn as_slice_i64(&self) -> Result<&[i64], CoreError>;
    fn as_any(&self) -> &dyn Any;
}

/// Generic CPU host tensor
#[derive(Clone, Debug, PartialEq)]
pub struct CpuTensor<T> {
    shape: TensorShape,
    dtype: DataType,
    data: Vec<T>,
}

impl<T: Clone + Send + Sync + 'static> CpuTensor<T> {
    pub fn new(shape: TensorShape, dtype: DataType, data: Vec<T>) -> Result<Self, CoreError> {
        if data.len() != shape.element_count() {
            return Err(CoreError::InvalidShape(format!(
                "Element count mismatch: shape requires {} elements, got {}",
                shape.element_count(),
                data.len()
            )));
        }
        Ok(Self { shape, dtype, data })
    }

    #[inline]
    pub fn shape(&self) -> &TensorShape {
        &self.shape
    }

    #[inline]
    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    #[inline]
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
}

impl CpuTensor<f32> {
    pub fn from_f32(shape: TensorShape, data: Vec<f32>) -> Result<Self, CoreError> {
        Self::new(shape, DataType::F32, data)
    }
}

impl CpuTensor<u8> {
    pub fn from_u8(shape: TensorShape, data: Vec<u8>) -> Result<Self, CoreError> {
        Self::new(shape, DataType::U8, data)
    }
}

impl CpuTensor<i32> {
    pub fn from_i32(shape: TensorShape, data: Vec<i32>) -> Result<Self, CoreError> {
        Self::new(shape, DataType::I32, data)
    }
}

impl CpuTensor<i64> {
    pub fn from_i64(shape: TensorShape, data: Vec<i64>) -> Result<Self, CoreError> {
        Self::new(shape, DataType::I64, data)
    }
}

impl AnyHostTensor for CpuTensor<f32> {
    fn shape(&self) -> &TensorShape {
        &self.shape
    }
    fn dtype(&self) -> DataType {
        self.dtype
    }
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const u8,
                self.data.len() * std::mem::size_of::<f32>(),
            )
        }
    }
    fn as_slice_f32(&self) -> Result<&[f32], CoreError> {
        Ok(&self.data)
    }
    fn as_slice_u8(&self) -> Result<&[u8], CoreError> {
        Err(CoreError::InvalidDataType {
            expected: DataType::U8,
            actual: self.dtype,
        })
    }
    fn as_slice_i32(&self) -> Result<&[i32], CoreError> {
        Err(CoreError::InvalidDataType {
            expected: DataType::I32,
            actual: self.dtype,
        })
    }
    fn as_slice_i64(&self) -> Result<&[i64], CoreError> {
        Err(CoreError::InvalidDataType {
            expected: DataType::I64,
            actual: self.dtype,
        })
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl AnyHostTensor for CpuTensor<u8> {
    fn shape(&self) -> &TensorShape {
        &self.shape
    }
    fn dtype(&self) -> DataType {
        self.dtype
    }
    fn as_bytes(&self) -> &[u8] {
        &self.data
    }
    fn as_slice_f32(&self) -> Result<&[f32], CoreError> {
        Err(CoreError::InvalidDataType {
            expected: DataType::F32,
            actual: self.dtype,
        })
    }
    fn as_slice_u8(&self) -> Result<&[u8], CoreError> {
        Ok(&self.data)
    }
    fn as_slice_i32(&self) -> Result<&[i32], CoreError> {
        Err(CoreError::InvalidDataType {
            expected: DataType::I32,
            actual: self.dtype,
        })
    }
    fn as_slice_i64(&self) -> Result<&[i64], CoreError> {
        Err(CoreError::InvalidDataType {
            expected: DataType::I64,
            actual: self.dtype,
        })
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl AnyHostTensor for CpuTensor<i32> {
    fn shape(&self) -> &TensorShape {
        &self.shape
    }
    fn dtype(&self) -> DataType {
        self.dtype
    }
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const u8,
                self.data.len() * std::mem::size_of::<i32>(),
            )
        }
    }
    fn as_slice_f32(&self) -> Result<&[f32], CoreError> {
        Err(CoreError::InvalidDataType {
            expected: DataType::F32,
            actual: self.dtype,
        })
    }
    fn as_slice_u8(&self) -> Result<&[u8], CoreError> {
        Err(CoreError::InvalidDataType {
            expected: DataType::U8,
            actual: self.dtype,
        })
    }
    fn as_slice_i32(&self) -> Result<&[i32], CoreError> {
        Ok(&self.data)
    }
    fn as_slice_i64(&self) -> Result<&[i64], CoreError> {
        Err(CoreError::InvalidDataType {
            expected: DataType::I64,
            actual: self.dtype,
        })
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl AnyHostTensor for CpuTensor<i64> {
    fn shape(&self) -> &TensorShape {
        &self.shape
    }
    fn dtype(&self) -> DataType {
        self.dtype
    }
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const u8,
                self.data.len() * std::mem::size_of::<i64>(),
            )
        }
    }
    fn as_slice_f32(&self) -> Result<&[f32], CoreError> {
        Err(CoreError::InvalidDataType {
            expected: DataType::F32,
            actual: self.dtype,
        })
    }
    fn as_slice_u8(&self) -> Result<&[u8], CoreError> {
        Err(CoreError::InvalidDataType {
            expected: DataType::U8,
            actual: self.dtype,
        })
    }
    fn as_slice_i32(&self) -> Result<&[i32], CoreError> {
        Err(CoreError::InvalidDataType {
            expected: DataType::I32,
            actual: self.dtype,
        })
    }
    fn as_slice_i64(&self) -> Result<&[i64], CoreError> {
        Ok(&self.data)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Opaque device-resident tensor buffer
pub trait TensorBuffer: Send + Sync + std::fmt::Debug {
    fn shape(&self) -> &TensorShape;
    fn dtype(&self) -> DataType;
    fn device(&self) -> &Device;
    fn byte_size(&self) -> usize {
        self.shape().byte_size(self.dtype())
    }

    /// Explicit readback into CPU host memory
    fn read_to_cpu(&self) -> Result<Box<dyn AnyHostTensor>, CoreError>;

    /// Explicit transfer to another execution device
    fn copy_to_device(&self, target: &Device) -> Result<Box<dyn TensorBuffer>, CoreError>;
}

/// Low-level device buffer abstraction
pub trait DeviceBuffer: Send + Sync + std::fmt::Debug {
    fn device(&self) -> &Device;
    fn byte_size(&self) -> usize;
    fn copy_to_host(&self, out: &mut [u8]) -> Result<(), CoreError>;
    fn copy_to_device(&self, target: &Device) -> Result<Box<dyn DeviceBuffer>, CoreError>;
}

impl TensorBuffer for CpuTensor<f32> {
    fn shape(&self) -> &TensorShape {
        &self.shape
    }
    fn dtype(&self) -> DataType {
        self.dtype
    }
    fn device(&self) -> &Device {
        static CPU: std::sync::OnceLock<Device> = std::sync::OnceLock::new();
        CPU.get_or_init(Device::cpu)
    }
    fn read_to_cpu(&self) -> Result<Box<dyn AnyHostTensor>, CoreError> {
        Ok(Box::new(self.clone()))
    }
    fn copy_to_device(&self, target: &Device) -> Result<Box<dyn TensorBuffer>, CoreError> {
        if target.is_cpu() {
            Ok(Box::new(self.clone()))
        } else {
            Err(CoreError::BufferTransferFailed(format!(
                "Transfer from CPU to {} not supported by pure CPU tensor; use device context allocator",
                target
            )))
        }
    }
}

impl TensorBuffer for CpuTensor<u8> {
    fn shape(&self) -> &TensorShape {
        &self.shape
    }
    fn dtype(&self) -> DataType {
        self.dtype
    }
    fn device(&self) -> &Device {
        static CPU: std::sync::OnceLock<Device> = std::sync::OnceLock::new();
        CPU.get_or_init(Device::cpu)
    }
    fn read_to_cpu(&self) -> Result<Box<dyn AnyHostTensor>, CoreError> {
        Ok(Box::new(self.clone()))
    }
    fn copy_to_device(&self, target: &Device) -> Result<Box<dyn TensorBuffer>, CoreError> {
        if target.is_cpu() {
            Ok(Box::new(self.clone()))
        } else {
            Err(CoreError::BufferTransferFailed(format!(
                "Transfer from CPU to {} not supported by pure CPU tensor; use device context allocator",
                target
            )))
        }
    }
}
