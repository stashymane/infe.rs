pub use processing_core::{FitMode, ImageFormat, ProcessingOptions, Rotation, TensorLayout};

use crate::error::CoreError;

/// Device-resident image accepted by a typed image processor.
pub trait DeviceImage: Send + Sync {
    fn width(&self) -> u32;
    fn height(&self) -> u32;
    fn format(&self) -> ImageFormat;
}

/// Host-resident image bytes (camera frame, decoded file, etc.).
#[derive(Clone, Debug, PartialEq)]
pub struct HostImage {
    width: u32,
    height: u32,
    format: ImageFormat,
    data: Vec<u8>,
}

impl HostImage {
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
}

impl DeviceImage for HostImage {
    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn format(&self) -> ImageFormat {
        self.format
    }
}
