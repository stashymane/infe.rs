use crate::sealed::Sealed;
use crate::image::{CpuImage, HardwareImage, Image};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    Cpu,
    Gpu,
    Npu,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DeviceInfo {
    pub kind: DeviceKind,
    pub id: usize,
    pub name: String,
}

impl std::fmt::Display for DeviceInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

/// Logical CPU execution device (zero-sized; host storage lives in [`HostTensor`](crate::HostTensor)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Cpu;

static CPU_INFO: std::sync::OnceLock<DeviceInfo> = std::sync::OnceLock::new();

impl Cpu {
    #[inline]
    pub fn info() -> &'static DeviceInfo {
        CPU_INFO.get_or_init(|| DeviceInfo {
            kind: DeviceKind::Cpu,
            id: 0,
            name: "CPU".to_string(),
        })
    }
}

/// Execution device handle. Each implementor owns its context and storage types.
pub trait Device: Sealed + Clone + Send + Sync + 'static {
    type Storage: Send + Sync + std::fmt::Debug + Clone;
    /// Materialized image type for this device (implements [`Image<Self>`]).
    type Image: Image<Self>;

    fn info(&self) -> &DeviceInfo;

    fn store(
        &self,
        shape: &crate::TensorShape,
        dtype: crate::DataType,
        bytes: &[u8],
    ) -> Result<Self::Storage, crate::CoreError>;

    fn load(
        &self,
        storage: &Self::Storage,
        shape: &crate::TensorShape,
        dtype: crate::DataType,
    ) -> Result<crate::HostTensor, crate::CoreError>;

    fn materialize_deferred(
        device: &Self,
        hardware: Arc<HardwareImage>,
    ) -> Result<Self::Image, crate::CoreError>;
}

impl Device for Cpu {
    type Storage = crate::tensor::HostBytes;
    type Image = CpuImage;

    fn info(&self) -> &DeviceInfo {
        Cpu::info()
    }

    fn store(
        &self,
        shape: &crate::TensorShape,
        dtype: crate::DataType,
        bytes: &[u8],
    ) -> Result<Self::Storage, crate::CoreError> {
        if bytes.len() != shape.byte_size(dtype) {
            return Err(crate::CoreError::InvalidShape(format!(
                "Byte size mismatch: expected {}, got {}",
                shape.byte_size(dtype),
                bytes.len()
            )));
        }
        Ok(crate::tensor::HostBytes::from_vec(bytes.to_vec()))
    }

    fn load(
        &self,
        storage: &Self::Storage,
        shape: &crate::TensorShape,
        dtype: crate::DataType,
    ) -> Result<crate::HostTensor, crate::CoreError> {
        crate::HostTensor::new(shape.clone(), dtype, storage.arc())
    }

    fn materialize_deferred(
        _device: &Self,
        hardware: Arc<HardwareImage>,
    ) -> Result<Self::Image, crate::CoreError> {
        Ok(CpuImage::from_hardware(hardware))
    }
}
