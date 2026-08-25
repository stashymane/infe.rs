use crate::device::cpu_device;
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
    /// Build a shape from dimensions, rejecting an empty list or any zero extent.
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

    /// Total number of elements, saturating rather than wrapping so that an
    /// absurd shape cannot produce a small size that then under-allocates.
    #[inline]
    pub fn element_count(&self) -> usize {
        self.dims
            .iter()
            .copied()
            .fold(1usize, |acc, d| acc.saturating_mul(d))
    }

    #[inline]
    pub fn byte_size(&self, dtype: DataType) -> usize {
        self.element_count().saturating_mul(dtype.element_size())
    }
}

impl TryFrom<Vec<usize>> for TensorShape {
    type Error = CoreError;

    fn try_from(dims: Vec<usize>) -> Result<Self, Self::Error> {
        Self::new(dims)
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

/// Opaque device-resident tensor buffer
pub trait TensorBuffer: Send + Sync + std::fmt::Debug {
    fn shape(&self) -> &TensorShape;
    fn dtype(&self) -> DataType;
    fn device(&self) -> &Device;
    fn byte_size(&self) -> usize {
        self.shape().byte_size(self.dtype())
    }

    /// Downcast support for device-specific buffer types.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Explicit readback into CPU host memory
    fn read_to_cpu(&self) -> Result<Box<dyn AnyHostTensor>, CoreError>;

    /// Explicit transfer to another execution device
    fn copy_to_device(&self, target: &Device) -> Result<Box<dyn TensorBuffer>, CoreError>;
}

macro_rules! impl_any_host_tensor_slice {
    ($self:ident, f32, f32) => {
        Ok(&$self.data)
    };
    ($self:ident, f32, $other:ident) => {
        Err(CoreError::InvalidDataType {
            expected: DataType::F32,
            actual: $self.dtype,
        })
    };
    ($self:ident, u8, u8) => {
        Ok(&$self.data)
    };
    ($self:ident, u8, $other:ident) => {
        Err(CoreError::InvalidDataType {
            expected: DataType::U8,
            actual: $self.dtype,
        })
    };
    ($self:ident, i32, i32) => {
        Ok(&$self.data)
    };
    ($self:ident, i32, $other:ident) => {
        Err(CoreError::InvalidDataType {
            expected: DataType::I32,
            actual: $self.dtype,
        })
    };
    ($self:ident, i64, i64) => {
        Ok(&$self.data)
    };
    ($self:ident, i64, $other:ident) => {
        Err(CoreError::InvalidDataType {
            expected: DataType::I64,
            actual: $self.dtype,
        })
    };
}

macro_rules! impl_any_host_tensor {
    (u8) => {
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
                impl_any_host_tensor_slice!(self, f32, u8)
            }
            fn as_slice_u8(&self) -> Result<&[u8], CoreError> {
                impl_any_host_tensor_slice!(self, u8, u8)
            }
            fn as_slice_i32(&self) -> Result<&[i32], CoreError> {
                impl_any_host_tensor_slice!(self, i32, u8)
            }
            fn as_slice_i64(&self) -> Result<&[i64], CoreError> {
                impl_any_host_tensor_slice!(self, i64, u8)
            }
        }
    };
    ($ty:ty, $native:ident, $safety:literal) => {
        impl AnyHostTensor for CpuTensor<$ty> {
            fn shape(&self) -> &TensorShape {
                &self.shape
            }
            fn dtype(&self) -> DataType {
                self.dtype
            }
            fn as_bytes(&self) -> &[u8] {
                // SAFETY: $safety
                unsafe {
                    std::slice::from_raw_parts(
                        self.data.as_ptr() as *const u8,
                        std::mem::size_of_val(self.data.as_slice()),
                    )
                }
            }
            fn as_slice_f32(&self) -> Result<&[f32], CoreError> {
                impl_any_host_tensor_slice!(self, f32, $native)
            }
            fn as_slice_u8(&self) -> Result<&[u8], CoreError> {
                impl_any_host_tensor_slice!(self, u8, $native)
            }
            fn as_slice_i32(&self) -> Result<&[i32], CoreError> {
                impl_any_host_tensor_slice!(self, i32, $native)
            }
            fn as_slice_i64(&self) -> Result<&[i64], CoreError> {
                impl_any_host_tensor_slice!(self, i64, $native)
            }
        }
    };
}

macro_rules! impl_cpu_tensor_buffer {
    ($ty:ty) => {
        impl TensorBuffer for CpuTensor<$ty> {
            fn shape(&self) -> &TensorShape {
                &self.shape
            }
            fn dtype(&self) -> DataType {
                self.dtype
            }
            fn device(&self) -> &Device {
                cpu_device()
            }
            fn as_any(&self) -> &dyn std::any::Any {
                self
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
    };
}

impl_any_host_tensor!(
    f32,
    f32,
    "`f32` has no padding or invalid bit patterns, so its bytes are always initialised and readable as `u8`. The length is exactly the vector's byte length and `u8` alignment (1) is weaker than `f32`'s, so the resulting slice stays inside the same allocation and borrows it for `&self`."
);
impl_any_host_tensor!(
    i32,
    i32,
    "as for the `f32` impl above; `i32` is likewise plain data with stronger alignment than `u8`."
);
impl_any_host_tensor!(
    i64,
    i64,
    "as for the `f32` impl above; `i64` is likewise plain data with stronger alignment than `u8`."
);
impl_any_host_tensor!(u8);

impl_cpu_tensor_buffer!(f32);
impl_cpu_tensor_buffer!(u8);
impl_cpu_tensor_buffer!(i32);
impl_cpu_tensor_buffer!(i64);
