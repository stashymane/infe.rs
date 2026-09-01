use infers_core::CoreError;
use fast_image_resize::images::{Image, ImageRef};
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use image::imageops;
use image::RgbImage;
use infers_core::{
    Cpu, HostImage, HostTensor, ImageFormat, ProcessingOptions, Rotation, Tensor, TensorShape,
};
use parking_lot::Mutex;
use processing_core::{FitMode, TensorLayout};

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

  pub fn process(
        &self,
        input: &HostImage,
        options: &ProcessingOptions,
    ) -> Result<Tensor<Cpu>, CoreError> {
        let src_bytes = input.as_bytes();
        if input.format() != ImageFormat::Rgb888 {
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
        pack_rgb_tensor(
            &dst_rgb_bytes,
            dest_w,
            dest_h,
            options.dest_format,
            options.dest_layout,
        )
    }
}

fn pack_rgb_tensor(
    interleaved: &[u8],
    dest_w: u32,
    dest_h: u32,
    dest_format: ImageFormat,
    dest_layout: TensorLayout,
) -> Result<Tensor<Cpu>, CoreError> {
    let hw = (dest_h as usize) * (dest_w as usize);
    let shape = match dest_layout {
        TensorLayout::Nhwc => TensorShape::new([1, dest_h as usize, dest_w as usize, 3])?,
        TensorLayout::Nchw => TensorShape::new([1, 3, dest_h as usize, dest_w as usize])?,
    };

    match dest_format {
        ImageFormat::Rgb888 => {
            if dest_layout == TensorLayout::Nhwc {
                let host = HostTensor::from_u8(shape, interleaved.to_vec())?;
                Tensor::from_host(&Cpu, &host)
            } else {
                let mut planar = vec![0u8; hw * 3];
                for y in 0..dest_h as usize {
                    for x in 0..dest_w as usize {
                        let i = y * dest_w as usize + x;
                        let base = i * 3;
                        planar[i] = interleaved[base];
                        planar[hw + i] = interleaved[base + 1];
                        planar[2 * hw + i] = interleaved[base + 2];
                    }
                }
                let host = HostTensor::from_u8(shape, planar)?;
                Tensor::from_host(&Cpu, &host)
            }
        }
        ImageFormat::Rgbf32 => {
            if dest_layout == TensorLayout::Nhwc {
                let mut f32_data = Vec::with_capacity(interleaved.len());
                for &b in interleaved {
                    f32_data.push((b as f32) / 255.0);
                }
                let host = HostTensor::from_f32(shape, f32_data)?;
                Tensor::from_host(&Cpu, &host)
            } else {
                let mut planar = vec![0.0f32; hw * 3];
                for y in 0..dest_h as usize {
                    for x in 0..dest_w as usize {
                        let i = y * dest_w as usize + x;
                        let base = i * 3;
                        planar[i] = interleaved[base] as f32 / 255.0;
                        planar[hw + i] = interleaved[base + 1] as f32 / 255.0;
                        planar[2 * hw + i] = interleaved[base + 2] as f32 / 255.0;
                    }
                }
                let host = HostTensor::from_f32(shape, planar)?;
                Tensor::from_host(&Cpu, &host)
            }
        }
        ImageFormat::Nv12 | ImageFormat::I420 => Err(CoreError::InvalidImageBuffer(
            "CpuImageProcessor does not support YUV destination".into(),
        )),
    }
}
