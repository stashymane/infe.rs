#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct ProcessingOptions {
    pub src_w: u32,
    pub src_h: u32,
    pub crop_x: u32,
    pub crop_y: u32,
    pub crop_w: u32,
    pub crop_h: u32,
    pub dest_w: u32,
    pub dest_h: u32,
    pub src_format: ImageFormat,
    pub dest_format: ImageFormat,
    pub fit_mode: FitMode,
    pub rotation: Rotation,
}

impl ProcessingOptions {
    /// Returns the effective crop region `(crop_x, crop_y, crop_w, crop_h)`.
    /// If `crop_w` or `crop_h` is 0, the full source dimensions are used.
    #[inline]
    pub fn effective_crop(&self) -> (u32, u32, u32, u32) {
        let cw = if self.crop_w == 0 { self.src_w } else { self.crop_w };
        let ch = if self.crop_h == 0 { self.src_h } else { self.crop_h };
        (self.crop_x, self.crop_y, cw, ch)
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
pub enum FitMode {
    #[default]
    STRETCH = 0,
    CONTAIN = 1,
    CROP = 2,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
pub enum ImageFormat {
    #[default]
    RGB888 = 0,
    RGBF32 = 1,
    /// 4:2:0 semi-planar: full-res Y, then interleaved UV (Android camera / `AHARDWAREBUFFER_FORMAT_Y8Cb8Cr8_420`).
    NV12 = 2,
    /// 4:2:0 planar: Y, then U, then V.
    I420 = 3,
}

impl ImageFormat {
    #[inline]
    pub fn is_yuv(self) -> bool {
        matches!(self, Self::NV12 | Self::I420)
    }

    /// Packed size in bytes of a `width` × `height` frame.
    #[inline]
    pub fn frame_bytes(self, width: u32, height: u32) -> u32 {
        match self {
            Self::RGB888 => width.saturating_mul(height).saturating_mul(3),
            Self::RGBF32 => width.saturating_mul(height).saturating_mul(12),
            Self::NV12 | Self::I420 => {
                let y = width.saturating_mul(height);
                y.saturating_add(y / 2)
            }
        }
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
pub enum Rotation {
    #[default]
    None = 0,
    R90DEG = 1,
    R180DEG = 2,
    R270DEG = 3,
}
