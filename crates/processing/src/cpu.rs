use infers_core::CoreError;
use fast_image_resize::images::{Image, ImageRef};
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use image::imageops;
use image::RgbImage;
use infers_core::{
    CpuTensor, ImageFormat, ImageInputBuffer, ProcessingOptions, Rotation, TensorBuffer,
    TensorShape,
};
use parking_lot::Mutex;
use processing_core::FitMode;

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
    #[must_use]
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
    ) -> Result<Box<dyn TensorBuffer>, CoreError> {
        let src_bytes = input.as_bytes().ok_or_else(|| {
            CoreError::InvalidImageBuffer("Input buffer does not contain CPU-accessible bytes".into())
        })?;

        // The pixel loops below step through `src_bytes` three bytes at a time,
        // so any other source layout would read the wrong pixels (or past the
        // end of the buffer).
        if input.format() != ImageFormat::Rgb888 {
            return Err(CoreError::InvalidImageBuffer(format!(
                "CpuImageProcessor only supports RGB888 input, got {:?}; use GpuImageProcessor",
                input.format()
            )));
        }
        if options.dest_format.is_yuv() {
            return Err(CoreError::InvalidImageBuffer(
                "CpuImageProcessor does not support YUV destination".into(),
            ));
        }

        let src_w = input.width();
        let src_h = input.height();

        let expected = ImageFormat::Rgb888.frame_bytes(src_w, src_h) as usize;
        if src_bytes.len() < expected {
            return Err(CoreError::InvalidImageBuffer(format!(
                "Input buffer holds {} bytes but {}x{} RGB888 needs {}",
                src_bytes.len(),
                src_w,
                src_h,
                expected
            )));
        }

        let (crop_x, crop_y, crop_w, crop_h) = options.effective_crop();

        if crop_w == 0 || crop_h == 0 {
            return Err(CoreError::InvalidImageBuffer(
                "Effective crop width and height must be greater than zero".into(),
            ));
        }

        // Checked arithmetic: `crop_x + crop_w` would otherwise wrap and let an
        // out-of-bounds region pass this validation.
        let exceeds = crop_x.checked_add(crop_w).is_none_or(|r| r > src_w)
            || crop_y.checked_add(crop_h).is_none_or(|b| b > src_h);
        if exceeds {
            return Err(CoreError::InvalidImageBuffer(format!(
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
            Rotation::Rot90 => imageops::rotate90(&cropped_img),
            Rotation::Rot180 => imageops::rotate180(&cropped_img),
            Rotation::Rot270 => imageops::rotate270(&cropped_img),
        };

        let (cur_w, cur_h) = (rotated_img.width(), rotated_img.height());
        let (dest_w, dest_h) = (options.dest_w, options.dest_h);

        if dest_w == 0 || dest_h == 0 {
            return Err(CoreError::InvalidImageBuffer(
                "Destination dimensions must be non-zero".into(),
            ));
        }

        // 3. Handle fit mode and scale using fast_image_resize
        let dst_rgb_bytes = match options.fit_mode {
            FitMode::Stretch => {
                let src_ref = ImageRef::new(cur_w, cur_h, rotated_img.as_raw(), PixelType::U8x3)
                    .map_err(|e| CoreError::ImageResizeFailed(e.to_string()))?;
                let mut dst_image = Image::new(dest_w, dest_h, PixelType::U8x3);

                let mut resizer = self.resizer.lock();
                resizer
                    .resize(
                        &src_ref,
                        &mut dst_image,
                        &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear)),
                    )
                    .map_err(|e| CoreError::ImageResizeFailed(e.to_string()))?;

                dst_image.into_vec()
            }
            FitMode::Contain => {
                let sx = dest_w as f64 / cur_w as f64;
                let sy = dest_h as f64 / cur_h as f64;
                let s = sx.min(sy);
                let scaled_w = (cur_w as f64 * s).round().max(1.0) as u32;
                let scaled_h = (cur_h as f64 * s).round().max(1.0) as u32;

                let src_ref = ImageRef::new(cur_w, cur_h, rotated_img.as_raw(), PixelType::U8x3)
                    .map_err(|e| CoreError::ImageResizeFailed(e.to_string()))?;
                let mut scaled_image = Image::new(scaled_w, scaled_h, PixelType::U8x3);

                let mut resizer = self.resizer.lock();
                resizer
                    .resize(
                        &src_ref,
                        &mut scaled_image,
                        &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear)),
                    )
                    .map_err(|e| CoreError::ImageResizeFailed(e.to_string()))?;

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
            FitMode::Crop => {
                let sx = dest_w as f64 / cur_w as f64;
                let sy = dest_h as f64 / cur_h as f64;
                let s = sx.max(sy);
                let scaled_w = (cur_w as f64 * s).round().max(1.0) as u32;
                let scaled_h = (cur_h as f64 * s).round().max(1.0) as u32;

                let src_ref = ImageRef::new(cur_w, cur_h, rotated_img.as_raw(), PixelType::U8x3)
                    .map_err(|e| CoreError::ImageResizeFailed(e.to_string()))?;
                let mut scaled_image = Image::new(scaled_w, scaled_h, PixelType::U8x3);

                let mut resizer = self.resizer.lock();
                resizer
                    .resize(
                        &src_ref,
                        &mut scaled_image,
                        &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear)),
                    )
                    .map_err(|e| CoreError::ImageResizeFailed(e.to_string()))?;

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
            ImageFormat::Rgb888 => {
                let tensor = CpuTensor::from_u8(shape, dst_rgb_bytes)?;
                Ok(Box::new(tensor))
            }
            ImageFormat::Rgbf32 => {
                let mut f32_data = Vec::with_capacity(dst_rgb_bytes.len());
                for b in dst_rgb_bytes {
                    f32_data.push((b as f32) / 255.0);
                }
                let tensor = CpuTensor::from_f32(shape, f32_data)?;
                Ok(Box::new(tensor))
            }
            ImageFormat::Nv12 | ImageFormat::I420 => Err(CoreError::InvalidImageBuffer(
                "CpuImageProcessor does not support YUV destination".into(),
            )),
        }
    }
}
