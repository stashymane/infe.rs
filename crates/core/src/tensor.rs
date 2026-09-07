use crate::device::{Cpu, Device};
use crate::error::CoreError;
use std::sync::Arc;

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
///
/// Bytes are reference-counted so CPU tensors and host views can share storage
/// without copying the payload.
#[derive(Clone, Debug, PartialEq)]
pub struct HostTensor {
    shape: TensorShape,
    dtype: DataType,
    bytes: Arc<[u8]>,
}

impl HostTensor {
    pub fn new(
        shape: TensorShape,
        dtype: DataType,
        bytes: impl Into<Arc<[u8]>>,
    ) -> Result<Self, CoreError> {
        let bytes = bytes.into();
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

    #[inline]
    pub fn arc_bytes(&self) -> Arc<[u8]> {
        Arc::clone(&self.bytes)
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

    /// Copy a byte range `[byte_offset, byte_offset + len)`.
    pub fn copy_bytes_range(&self, byte_offset: usize, len: usize) -> Result<Vec<u8>, CoreError> {
        let end = byte_offset.checked_add(len).ok_or_else(|| {
            CoreError::InvalidArgument("byte range overflow".into())
        })?;
        if end > self.bytes.len() {
            return Err(CoreError::InvalidArgument(format!(
                "byte range [{byte_offset}, {end}) exceeds tensor byte size {}",
                self.bytes.len()
            )));
        }
        Ok(self.bytes[byte_offset..end].to_vec())
    }

    /// Copy `len` f32 elements starting at `start_elem`.
    pub fn copy_f32_range(&self, start_elem: usize, len: usize) -> Result<Vec<f32>, CoreError> {
        let slice = self.as_slice_f32()?;
        copy_elem_range(slice, start_elem, len)
    }

    /// Copy `len` i32 elements starting at `start_elem`.
    pub fn copy_i32_range(&self, start_elem: usize, len: usize) -> Result<Vec<i32>, CoreError> {
        let slice = self.as_slice_i32()?;
        copy_elem_range(slice, start_elem, len)
    }

    /// Copy `len` i64 elements starting at `start_elem`.
    pub fn copy_i64_range(&self, start_elem: usize, len: usize) -> Result<Vec<i64>, CoreError> {
        let slice = self.as_slice_i64()?;
        copy_elem_range(slice, start_elem, len)
    }
}

fn copy_elem_range<T: Copy>(slice: &[T], start: usize, len: usize) -> Result<Vec<T>, CoreError> {
    let end = start.checked_add(len).ok_or_else(|| {
        CoreError::InvalidArgument("element range overflow".into())
    })?;
    if end > slice.len() {
        return Err(CoreError::InvalidArgument(format!(
            "element range [{start}, {end}) exceeds tensor element count {}",
            slice.len()
        )));
    }
    Ok(slice[start..end].to_vec())
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

impl Tensor<Cpu> {
    /// Adopt a host tensor by cloning the shared [`Arc`] payload (no byte copy).
    pub fn adopt_host(host: &HostTensor) -> Self {
        Self::from_storage(
            Cpu,
            host.shape().clone(),
            host.dtype(),
            HostBytes(host.arc_bytes()),
        )
    }

    /// Pointer and length of the underlying host bytes (valid while this tensor lives).
    pub fn host_ptr_len(&self) -> (*const u8, usize) {
        let slice = self.storage.as_slice();
        (slice.as_ptr(), slice.len())
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

/// CPU tensor storage is shared host bytes.
#[derive(Clone, Debug, PartialEq)]
pub struct HostBytes(pub Arc<[u8]>);

impl HostBytes {
    #[inline]
    pub fn from_vec(bytes: Vec<u8>) -> Self {
        Self(bytes.into())
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.0
    }

    #[inline]
    pub fn arc(&self) -> Arc<[u8]> {
        Arc::clone(&self.0)
    }
}

impl From<Vec<u8>> for HostBytes {
    fn from(bytes: Vec<u8>) -> Self {
        Self::from_vec(bytes)
    }
}

impl From<Arc<[u8]>> for HostBytes {
    fn from(bytes: Arc<[u8]>) -> Self {
        Self(bytes)
    }
}
