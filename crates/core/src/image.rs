pub use processing_core::{FitMode, ImageFormat, ProcessingOptions, Rotation};

use crate::device::Device;
use crate::error::CoreError;

/// Trait representing an input image for preprocessing
pub trait ImageInputBuffer: Send + Sync {
    fn width(&self) -> u32;
    fn height(&self) -> u32;
    fn format(&self) -> ImageFormat;
    fn device(&self) -> &Device;
    fn as_bytes(&self) -> Option<&[u8]>;
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
        let expected_bytes = match format {
            ImageFormat::RGB888 => (width as usize) * (height as usize) * 3,
            ImageFormat::RGBF32 => (width as usize) * (height as usize) * 3 * 4,
        };

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
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    #[inline]
    pub fn into_data(self) -> Vec<u8> {
        self.data
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
        static CPU: std::sync::OnceLock<Device> = std::sync::OnceLock::new();
        CPU.get_or_init(Device::cpu)
    }

    fn as_bytes(&self) -> Option<&[u8]> {
        Some(&self.data)
    }
}
