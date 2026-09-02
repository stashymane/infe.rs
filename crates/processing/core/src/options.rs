#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
pub enum TensorLayout {
    #[default]
    Nhwc = 0,
    Nchw = 1,
}

impl TensorLayout {
    /// Default output layout: NCHW for normalized f32 model input, NHWC for packed RGB888.
    pub fn default_for_dest_format(dest_format: ImageFormat) -> Self {
        match dest_format {
            ImageFormat::Rgbf32 => Self::Nchw,
            ImageFormat::Rgb888 | ImageFormat::Nv12 | ImageFormat::I420 => Self::Nhwc,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    pub dest_layout: TensorLayout,
}

impl Default for ProcessingOptions {
    fn default() -> Self {
        let dest_format = ImageFormat::default();
        Self {
            src_w: 0,
            src_h: 0,
            crop_x: 0,
            crop_y: 0,
            crop_w: 0,
            crop_h: 0,
            dest_w: 0,
            dest_h: 0,
            src_format: ImageFormat::default(),
            dest_format,
            fit_mode: FitMode::default(),
            rotation: Rotation::default(),
            dest_layout: TensorLayout::default_for_dest_format(dest_format),
        }
    }
}

impl ProcessingOptions {
    pub fn for_model_input() -> Self {
        let dest_format = ImageFormat::Rgbf32;
        Self {
            dest_format,
            dest_layout: TensorLayout::Nchw,
            ..Self::default()
        }
    }

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
    Stretch = 0,
    Contain = 1,
    Crop = 2,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
pub enum ImageFormat {
    #[default]
    Rgb888 = 0,
    Rgbf32 = 1,
    /// 4:2:0 semi-planar: full-res Y, then interleaved UV (Android camera / `AHARDWAREBUFFER_FORMAT_Y8Cb8Cr8_420`).
    Nv12 = 2,
    /// 4:2:0 planar: Y, then U, then V.
    I420 = 3,
}

impl ImageFormat {
    #[inline]
    pub fn is_yuv(self) -> bool {
        matches!(self, Self::Nv12 | Self::I420)
    }

    /// Packed size in bytes of a `width` × `height` frame.
    #[inline]
    pub fn frame_bytes(self, width: u32, height: u32) -> u32 {
        match self {
            Self::Rgb888 => width.saturating_mul(height).saturating_mul(3),
            Self::Rgbf32 => width.saturating_mul(height).saturating_mul(12),
            Self::Nv12 | Self::I420 => {
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
    Rot90 = 1,
    Rot180 = 2,
    Rot270 = 3,
}
