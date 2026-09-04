use crate::device::{Cpu, Device};
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

/// Owned host tensor payload (dtype-tagged bytes).
#[derive(Clone, Debug, PartialEq)]
pub struct HostTensor {
    shape: TensorShape,
    dtype: DataType,
    bytes: Vec<u8>,
}

impl HostTensor {
    pub fn new(shape: TensorShape, dtype: DataType, bytes: Vec<u8>) -> Result<Self, CoreError> {
        if bytes.len() != shape.byte_size(dtype) {
            return Err(CoreError::InvalidShape(format!(
                "Byte size mismatch: shape requires {} bytes, got {}",
                shape.byte_size(dtype),
                bytes.len()
            )));
        }
        Ok(Self {
            shape,
            dtype,
            bytes,
        })
    }

    pub fn from_f32(shape: TensorShape, data: Vec<f32>) -> Result<Self, CoreError> {
        let bytes = crate::bytes::vec_to_bytes(&data)?;
        Self::new(shape, DataType::F32, bytes)
    }

    pub fn from_u8(shape: TensorShape, data: Vec<u8>) -> Result<Self, CoreError> {
        Self::new(shape, DataType::U8, data)
    }

    pub fn from_i32(shape: TensorShape, data: Vec<i32>) -> Result<Self, CoreError> {
        let bytes = crate::bytes::vec_to_bytes(&data)?;
        Self::new(shape, DataType::I32, bytes)
    }

    pub fn from_i64(shape: TensorShape, data: Vec<i64>) -> Result<Self, CoreError> {
        let bytes = crate::bytes::vec_to_bytes(&data)?;
        Self::new(shape, DataType::I64, bytes)
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
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn as_slice_f32(&self) -> Result<&[f32], CoreError> {
        if self.dtype != DataType::F32 {
            return Err(CoreError::InvalidDataType {
                expected: DataType::F32,
                actual: self.dtype,
            });
        }
        crate::bytes::cast_bytes(&self.bytes)
    }

    pub fn as_slice_u8(&self) -> Result<&[u8], CoreError> {
        if self.dtype != DataType::U8 {
            return Err(CoreError::InvalidDataType {
                expected: DataType::U8,
                actual: self.dtype,
            });
        }
        Ok(&self.bytes)
    }

    pub fn as_slice_i32(&self) -> Result<&[i32], CoreError> {
        if self.dtype != DataType::I32 {
            return Err(CoreError::InvalidDataType {
                expected: DataType::I32,
                actual: self.dtype,
            });
        }
        crate::bytes::cast_bytes(&self.bytes)
    }

    pub fn as_slice_i64(&self) -> Result<&[i64], CoreError> {
        if self.dtype != DataType::I64 {
            return Err(CoreError::InvalidDataType {
                expected: DataType::I64,
                actual: self.dtype,
            });
        }
        crate::bytes::cast_bytes(&self.bytes)
    }
}

/// Device-resident tensor. The device type parameter prevents cross-device misuse at compile time.
#[derive(Clone, Debug)]
pub struct Tensor<D: Device> {
    device: D,
    shape: TensorShape,
    dtype: DataType,
    storage: D::Storage,
}

impl<D: Device> Tensor<D> {
    pub fn from_storage(
        device: D,
        shape: TensorShape,
        dtype: DataType,
        storage: D::Storage,
    ) -> Self {
        Self {
            device,
            shape,
            dtype,
            storage,
        }
    }

    pub fn from_host(device: &D, host: &HostTensor) -> Result<Self, CoreError> {
        let storage = device.store(host.shape(), host.dtype(), host.as_bytes())?;
        Ok(Self {
            device: device.clone(),
            shape: host.shape().clone(),
            dtype: host.dtype(),
            storage,
        })
    }

    #[inline]
    pub fn device(&self) -> &D {
        &self.device
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
    pub fn byte_size(&self) -> usize {
        self.shape.byte_size(self.dtype)
    }

    #[inline]
    pub fn storage(&self) -> &D::Storage {
        &self.storage
    }

    pub fn read_to_host(&self) -> Result<HostTensor, CoreError> {
        self.device.load(&self.storage, &self.shape, self.dtype)
    }

    /// Transfer to another device. The target handle carries any upload context.
    pub fn to_device<T: TensorAdopt>(&self, target: &T) -> Result<Tensor<T>, CoreError> {
        if let Some(adopted) = target.try_adopt_tensor(self) {
            return Ok(adopted);
        }
        Tensor::from_host(target, &self.read_to_host()?)
    }
}

/// Optional fast-path adoption of a tensor already on an equivalent device handle.
pub trait TensorAdopt: Device {
    fn try_adopt_tensor<S: Device>(&self, tensor: &Tensor<S>) -> Option<Tensor<Self>>;
}

impl TensorAdopt for Cpu {
    fn try_adopt_tensor<S: Device>(&self, tensor: &Tensor<S>) -> Option<Tensor<Self>> {
        if std::any::TypeId::of::<S>() != std::any::TypeId::of::<Cpu>() {
            return None;
        }
        // SAFETY: S is Cpu when the TypeIds match.
        let cpu_tensor = unsafe { &*(tensor as *const Tensor<S> as *const Tensor<Cpu>) };
        Some(Tensor {
            device: Cpu,
            shape: cpu_tensor.shape.clone(),
            dtype: cpu_tensor.dtype,
            storage: cpu_tensor.storage.clone(),
        })
    }
}

/// CPU tensor storage is host bytes.
#[derive(Clone, Debug, PartialEq)]
pub struct HostBytes(pub Vec<u8>);
