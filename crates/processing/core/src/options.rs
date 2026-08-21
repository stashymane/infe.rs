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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FitMode {
    STRETCH = 0,
    CONTAIN = 1,
    CROP = 2,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageFormat {
    RGB888 = 0,
    RGBF32 = 1,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Rotation {
    None = 0,
    R90DEG = 1,
    R180DEG = 2,
    R270DEG = 3,
}
