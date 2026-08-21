use crate::error::ProcessingError;
use fast_image_resize::images::{Image, ImageRef};
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use image::imageops;
use image::RgbImage;
use infers_core::{
    AnyHostTensor, CpuTensor, DataType, Device, ImageFormat, ImageInputBuffer, ProcessingOptions,
    Rotation, TensorBuffer, TensorShape,
};
use processing_core::FitMode;
use std::sync::Mutex;

/// High-performance CPU image processor delegating to `fast_image_resize` (SIMD) and `image`
pub struct CpuImageProcessor {
    resizer: Mutex<Resizer>,
}

impl Default for CpuImageProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuImageProcessor {
    pub fn new() -> Self {
        Self {
            resizer: Mutex::new(Resizer::new()),
        }
    }

    /// Process an input image using CPU SIMD resizing and transformations
    pub fn process(
        &self,
        input: &dyn ImageInputBuffer,
        options: &ProcessingOptions,
    ) -> Result<Box<dyn TensorBuffer>, ProcessingError> {
        let src_bytes = input.as_bytes().ok_or_else(|| {
            ProcessingError::InvalidBuffer("Input buffer does not contain CPU-accessible bytes".into())
        })?;

        let src_w = input.width();
        let src_h = input.height();
        let (crop_x, crop_y, crop_w, crop_h) = options.effective_crop();

        if crop_w == 0 || crop_h == 0 {
            return Err(ProcessingError::InvalidBuffer(
                "Effective crop width and height must be greater than zero".into(),
            ));
        }

        if crop_x + crop_w > src_w || crop_y + crop_h > src_h {
            return Err(ProcessingError::InvalidBuffer(format!(
                "Crop region ({}, {}, {}, {}) exceeds source image dimensions ({}x{})",
                crop_x, crop_y, crop_w, crop_h, src_w, src_h
            )));
        }

        // 1. Extract crop region as RgbImage
        let mut cropped_img = RgbImage::new(crop_w, crop_h);
        for y in 0..crop_h {
            let src_row_start = ((crop_y + y) * src_w + crop_x) as usize * 3;
            let src_row_end = src_row_start + (crop_w as usize * 3);
            let dst_row_start = (y * crop_w) as usize * 3;
            let dst_row_end = dst_row_start + (crop_w as usize * 3);
            cropped_img.as_mut()[dst_row_start..dst_row_end]
                .copy_from_slice(&src_bytes[src_row_start..src_row_end]);
        }

        // 2. Apply rotation if requested
        let rotated_img = match options.rotation {
            Rotation::None => cropped_img,
            Rotation::R90DEG => imageops::rotate90(&cropped_img),
            Rotation::R180DEG => imageops::rotate180(&cropped_img),
            Rotation::R270DEG => imageops::rotate270(&cropped_img),
        };

        let (cur_w, cur_h) = (rotated_img.width(), rotated_img.height());
        let (dest_w, dest_h) = (options.dest_w, options.dest_h);

        if dest_w == 0 || dest_h == 0 {
            return Err(ProcessingError::InvalidBuffer(
                "Destination dimensions must be non-zero".into(),
            ));
        }

        // 3. Handle fit mode and scale using fast_image_resize
        let dst_rgb_bytes = match options.fit_mode {
            FitMode::STRETCH => {
                let src_ref = ImageRef::new(cur_w, cur_h, rotated_img.as_raw(), PixelType::U8x3)
                    .map_err(|e| ProcessingError::ResizeError(e.to_string()))?;
                let mut dst_image = Image::new(dest_w, dest_h, PixelType::U8x3);

                let mut resizer = self.resizer.lock().unwrap();
                resizer
                    .resize(
                        &src_ref,
                        &mut dst_image,
                        &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear)),
                    )
                    .map_err(|e| ProcessingError::ResizeError(e.to_string()))?;

                dst_image.into_vec()
            }
            FitMode::CONTAIN => {
                let sx = dest_w as f64 / cur_w as f64;
                let sy = dest_h as f64 / cur_h as f64;
                let s = sx.min(sy);
                let scaled_w = (cur_w as f64 * s).round().max(1.0) as u32;
                let scaled_h = (cur_h as f64 * s).round().max(1.0) as u32;

                let src_ref = ImageRef::new(cur_w, cur_h, rotated_img.as_raw(), PixelType::U8x3)
                    .map_err(|e| ProcessingError::ResizeError(e.to_string()))?;
                let mut scaled_image = Image::new(scaled_w, scaled_h, PixelType::U8x3);

                let mut resizer = self.resizer.lock().unwrap();
                resizer
                    .resize(
                        &src_ref,
                        &mut scaled_image,
                        &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear)),
                    )
                    .map_err(|e| ProcessingError::ResizeError(e.to_string()))?;

                let scaled_bytes = scaled_image.into_vec();
                let mut final_buf = vec![0u8; (dest_w * dest_h * 3) as usize];

                let pad_x = (dest_w.saturating_sub(scaled_w)) / 2;
                let pad_y = (dest_h.saturating_sub(scaled_h)) / 2;

                for y in 0..scaled_h {
                    let src_offset = (y * scaled_w * 3) as usize;
                    let dst_offset = (((pad_y + y) * dest_w + pad_x) * 3) as usize;
                    let row_len = (scaled_w * 3) as usize;
                    final_buf[dst_offset..dst_offset + row_len]
                        .copy_from_slice(&scaled_bytes[src_offset..src_offset + row_len]);
                }

                final_buf
            }
            FitMode::CROP => {
                let sx = dest_w as f64 / cur_w as f64;
                let sy = dest_h as f64 / cur_h as f64;
                let s = sx.max(sy);
                let scaled_w = (cur_w as f64 * s).round().max(1.0) as u32;
                let scaled_h = (cur_h as f64 * s).round().max(1.0) as u32;

                let src_ref = ImageRef::new(cur_w, cur_h, rotated_img.as_raw(), PixelType::U8x3)
                    .map_err(|e| ProcessingError::ResizeError(e.to_string()))?;
                let mut scaled_image = Image::new(scaled_w, scaled_h, PixelType::U8x3);

                let mut resizer = self.resizer.lock().unwrap();
                resizer
                    .resize(
                        &src_ref,
                        &mut scaled_image,
                        &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear)),
                    )
                    .map_err(|e| ProcessingError::ResizeError(e.to_string()))?;

                let scaled_bytes = scaled_image.into_vec();
                let mut final_buf = vec![0u8; (dest_w * dest_h * 3) as usize];

                let crop_start_x = (scaled_w.saturating_sub(dest_w)) / 2;
                let crop_start_y = (scaled_h.saturating_sub(dest_h)) / 2;

                for y in 0..dest_h {
                    let src_offset = (((crop_start_y + y) * scaled_w + crop_start_x) * 3) as usize;
                    let dst_offset = (y * dest_w * 3) as usize;
                    let row_len = (dest_w * 3) as usize;
                    final_buf[dst_offset..dst_offset + row_len]
                        .copy_from_slice(&scaled_bytes[src_offset..src_offset + row_len]);
                }

                final_buf
            }
        };

        // 4. Format conversion into TensorBuffer
        let shape = TensorShape::new(vec![1, dest_h as usize, dest_w as usize, 3])?;
        match options.dest_format {
            ImageFormat::RGB888 => {
                let tensor = CpuTensor::from_u8(shape, dst_rgb_bytes)?;
                Ok(Box::new(CpuTensorBuffer::new(tensor)))
            }
            ImageFormat::RGBF32 => {
                let f32_data: Vec<f32> = dst_rgb_bytes
                    .into_iter()
                    .map(|b| (b as f32) / 255.0)
                    .collect();
                let tensor = CpuTensor::from_f32(shape, f32_data)?;
                Ok(Box::new(CpuTensorBuffer::new(tensor)))
            }
        }
    }
}

/// A `TensorBuffer` wrapper around a host CPU tensor
#[derive(Debug)]
pub struct CpuTensorBuffer<T> {
    device: Device,
    tensor: CpuTensor<T>,
}

impl<T: Clone + Send + Sync + 'static> CpuTensorBuffer<T> {
    pub fn new(tensor: CpuTensor<T>) -> Self {
        Self {
            device: Device::cpu(),
            tensor,
        }
    }
}

impl TensorBuffer for CpuTensorBuffer<f32> {
    fn shape(&self) -> &TensorShape {
        self.tensor.shape()
    }

    fn dtype(&self) -> DataType {
        self.tensor.dtype()
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn read_to_cpu(&self) -> Result<Box<dyn AnyHostTensor>, infers_core::CoreError> {
        Ok(Box::new(self.tensor.clone()))
    }

    fn copy_to_device(&self, target: &Device) -> Result<Box<dyn TensorBuffer>, infers_core::CoreError> {
        if target.is_cpu() {
            Ok(Box::new(self.clone()))
        } else {
            Err(infers_core::CoreError::BufferTransferFailed(format!(
                "Cannot copy CPU tensor to non-CPU device {} without backend context",
                target
            )))
        }
    }
}

impl Clone for CpuTensorBuffer<f32> {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            tensor: self.tensor.clone(),
        }
    }
}

impl TensorBuffer for CpuTensorBuffer<u8> {
    fn shape(&self) -> &TensorShape {
        self.tensor.shape()
    }

    fn dtype(&self) -> DataType {
        self.tensor.dtype()
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn read_to_cpu(&self) -> Result<Box<dyn AnyHostTensor>, infers_core::CoreError> {
        Ok(Box::new(self.tensor.clone()))
    }

    fn copy_to_device(&self, target: &Device) -> Result<Box<dyn TensorBuffer>, infers_core::CoreError> {
        if target.is_cpu() {
            Ok(Box::new(self.clone()))
        } else {
            Err(infers_core::CoreError::BufferTransferFailed(format!(
                "Cannot copy CPU tensor to non-CPU device {} without backend context",
                target
            )))
        }
    }
}

impl Clone for CpuTensorBuffer<u8> {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            tensor: self.tensor.clone(),
        }
    }
}

/// Software CPU reference shader emulator executing the exact SPIR-V `convert_main` math
pub fn reference_shader_convert_main(
    src: &[u8],
    options: &ProcessingOptions,
) -> Result<Vec<u8>, ProcessingError> {
    let (dest_w, dest_h) = (options.dest_w, options.dest_h);
    let bytes_per_pixel = match options.dest_format {
        ImageFormat::RGB888 => 3,
        ImageFormat::RGBF32 => 12,
    };
    let mut dst = vec![0u8; (dest_w * dest_h * bytes_per_pixel) as usize];

    for dy in 0..dest_h {
        for dx in 0..dest_w {
            let (sx, sy, valid) = map_dst_to_src(options, dx as f32, dy as f32);
            let (r, g, b) = if valid {
                sample_rgb_bilinear(src, options.src_w, options.src_h, sx, sy)
            } else {
                (0.0, 0.0, 0.0)
            };

            match options.dest_format {
                ImageFormat::RGB888 => {
                    let o = ((dy * dest_w + dx) * 3) as usize;
                    dst[o] = r.round().clamp(0.0, 255.0) as u8;
                    dst[o + 1] = g.round().clamp(0.0, 255.0) as u8;
                    dst[o + 2] = b.round().clamp(0.0, 255.0) as u8;
                }
                ImageFormat::RGBF32 => {
                    let base = ((dy * dest_w + dx) * 12) as usize;
                    let r_bits = (r / 255.0).to_bits();
                    dst[base..base + 4].copy_from_slice(&r_bits.to_le_bytes());
                    let g_bits = (g / 255.0).to_bits();
                    dst[base + 4..base + 8].copy_from_slice(&g_bits.to_le_bytes());
                    let b_bits = (b / 255.0).to_bits();
                    dst[base + 8..base + 12].copy_from_slice(&b_bits.to_le_bytes());
                }
            }
        }
    }

    Ok(dst)
}

fn bilinear(v00: f32, v10: f32, v01: f32, v11: f32, tx: f32, ty: f32) -> f32 {
    let v0 = v00 + (v10 - v00) * tx;
    let v1 = v01 + (v11 - v01) * tx;
    v0 + (v1 - v0) * ty
}

fn sample_rgb_bilinear(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    fx: f32,
    fy: f32,
) -> (f32, f32, f32) {
    if fx < 0.0 || fy < 0.0 || fx >= src_w as f32 || fy >= src_h as f32 {
        return (0.0, 0.0, 0.0);
    }
    let x0 = fx.floor();
    let y0 = fy.floor();
    let x1 = (x0 + 1.0).min((src_w - 1) as f32);
    let y1 = (y0 + 1.0).min((src_h - 1) as f32);
    let tx = fx - x0;
    let ty = fy - y0;
    let w = src_w as usize;
    let i00 = (y0 as usize * w + x0 as usize) * 3;
    let i10 = (y0 as usize * w + x1 as usize) * 3;
    let i01 = (y1 as usize * w + x0 as usize) * 3;
    let i11 = (y1 as usize * w + x1 as usize) * 3;

    if i11 + 2 >= src.len() {
        return (0.0, 0.0, 0.0);
    }

    let r = bilinear(
        src[i00] as f32,
        src[i10] as f32,
        src[i01] as f32,
        src[i11] as f32,
        tx,
        ty,
    );
    let g = bilinear(
        src[i00 + 1] as f32,
        src[i10 + 1] as f32,
        src[i01 + 1] as f32,
        src[i11 + 1] as f32,
        tx,
        ty,
    );
    let b = bilinear(
        src[i00 + 2] as f32,
        src[i10 + 2] as f32,
        src[i01 + 2] as f32,
        src[i11 + 2] as f32,
        tx,
        ty,
    );
    (r, g, b)
}

fn map_dst_to_src(params: &ProcessingOptions, dx: f32, dy: f32) -> (f32, f32, bool) {
    let (crop_x, crop_y, crop_w, crop_h) = {
        let (x, y, w, h) = params.effective_crop();
        (x as f32, y as f32, w as f32, h as f32)
    };

    let (rot_w, rot_h) = match params.rotation {
        Rotation::None | Rotation::R180DEG => (crop_w, crop_h),
        Rotation::R90DEG | Rotation::R270DEG => (crop_h, crop_w),
    };

    let dst_w = params.dest_w as f32;
    let dst_h = params.dest_h as f32;

    let (scale_x, scale_y, pad_x, pad_y) = match params.fit_mode {
        FitMode::STRETCH => (dst_w / rot_w, dst_h / rot_h, 0.0, 0.0),
        FitMode::CONTAIN | FitMode::CROP => {
            let sx = dst_w / rot_w;
            let sy = dst_h / rot_h;
            let s = if params.fit_mode == FitMode::CROP {
                if sx > sy { sx } else { sy }
            } else {
                if sx < sy { sx } else { sy }
            };
            let scaled_w = rot_w * s;
            let scaled_h = rot_h * s;
            let pad_x = (dst_w - scaled_w) * 0.5;
            let pad_y = (dst_h - scaled_h) * 0.5;
            (s, s, pad_x, pad_y)
        }
    };

    let rx = (dx - pad_x) / scale_x;
    let ry = (dy - pad_y) / scale_y;

    if rx < 0.0 || ry < 0.0 || rx >= rot_w || ry >= rot_h {
        return (0.0, 0.0, false);
    }

    let (cx, cy) = match params.rotation {
        Rotation::None => (rx, ry),
        Rotation::R90DEG => (ry, (crop_h - 1.0) - rx),
        Rotation::R180DEG => ((crop_w - 1.0) - rx, (crop_h - 1.0) - ry),
        Rotation::R270DEG => ((crop_w - 1.0) - ry, rx),
    };

    let sx = cx + crop_x;
    let sy = cy + crop_y;

    if sx < 0.0 || sy < 0.0 || sx >= params.src_w as f32 || sy >= params.src_h as f32 {
        return (0.0, 0.0, false);
    }

    (sx, sy, true)
}
