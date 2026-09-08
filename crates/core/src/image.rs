pub use processing_core::{FitMode, ImageFormat, ProcessingOptions, TensorLayout};

use crate::device::Device;
use crate::error::CoreError;
use std::sync::Arc;

/// Materialized image resident on device [D].
pub trait Image<D: Device>: Send + Sync {
    fn width(&self) -> u32;
    fn height(&self) -> u32;
    fn format(&self) -> ImageFormat;
}

/// Input-only source bytes (camera frame, decoded file, AHB CPU copy). Not processable directly.
#[derive(Debug)]
pub struct HardwareImage {
    width: u32,
    height: u32,
    format: ImageFormat,
    data: Vec<u8>,
}

impl PartialEq for HardwareImage {
    fn eq(&self, other: &Self) -> bool {
        self.width == other.width
            && self.height == other.height
            && self.format == other.format
            && self.data == other.data
    }
}

impl Clone for HardwareImage {
    fn clone(&self) -> Self {
        Self {
            width: self.width,
            height: self.height,
            format: self.format,
            data: self.data.clone(),
        }
    }
}

impl HardwareImage {
    pub fn new(
        width: u32,
        height: u32,
        format: ImageFormat,
        data: Vec<u8>,
    ) -> Result<Self, CoreError> {
        let expected_bytes = format.frame_bytes(width, height) as usize;

        if data.len() != expected_bytes {
            return Err(CoreError::InvalidArgument(format!(
                "Image data size mismatch: expected {} bytes, got {}",
                expected_bytes,
                data.len()
            )));
        }

        Ok(Self {
            width,
            height,
            format,
            data,
        })
    }

    #[inline]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[inline]
    pub fn height(&self) -> u32 {
        self.height
    }

    #[inline]
    pub fn format(&self) -> ImageFormat {
        self.format
    }

    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Defer placement on CPU (zero-copy wrap at materialize time).
    pub fn on_cpu(self) -> crate::Deferred<Cpu> {
        crate::Deferred::from_hardware(Cpu, Arc::new(self))
    }
}

use crate::Cpu;

/// Materialized CPU-resident image (wraps [`HardwareImage`] bytes without copy).
#[derive(Clone, Debug)]
pub struct CpuImage {
    inner: Arc<HardwareImage>,
}

impl CpuImage {
    pub(crate) fn from_hardware(hardware: Arc<HardwareImage>) -> Self {
        Self { inner: hardware }
    }

    pub fn hardware(&self) -> Arc<HardwareImage> {
        Arc::clone(&self.inner)
    }
}

impl Image<Cpu> for CpuImage {
    fn width(&self) -> u32 {
        self.inner.width()
    }

    fn height(&self) -> u32 {
        self.inner.height()
    }

    fn format(&self) -> ImageFormat {
        self.inner.format()
    }
}
