pub use processing_core::{FitMode, ImageFormat, ProcessingOptions, Rotation, TensorLayout};

use crate::device::cpu_device;
use crate::device::Device;
use crate::error::CoreError;
use std::any::Any;

/// Trait representing an input image for preprocessing
pub trait ImageInputBuffer: Send + Sync {
    fn width(&self) -> u32;
    fn height(&self) -> u32;
    fn format(&self) -> ImageFormat;
    fn device(&self) -> &Device;
    fn as_bytes(&self) -> Option<&[u8]>;
    fn as_any(&self) -> &dyn Any;
}

/// A CPU-resident image buffer
#[derive(Clone, Debug, PartialEq)]
pub struct CpuImageBuffer {
    width: u32,
    height: u32,
    format: ImageFormat,
    data: Vec<u8>,
}

impl CpuImageBuffer {
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

}

impl ImageInputBuffer for CpuImageBuffer {
    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn format(&self) -> ImageFormat {
        self.format
    }

    fn device(&self) -> &Device {
        cpu_device()
    }

    fn as_bytes(&self) -> Option<&[u8]> {
        Some(&self.data)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
