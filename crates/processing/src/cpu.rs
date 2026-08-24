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

        if input.format().is_yuv() {
            return Err(ProcessingError::InvalidBuffer(
                "CpuImageProcessor does not support YUV input; use GpuImageProcessor".into(),
            ));
        }
        if options.dest_format.is_yuv() {
            return Err(ProcessingError::InvalidBuffer(
                "CpuImageProcessor does not support YUV destination".into(),
            ));
        }

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
            ImageFormat::NV12 | ImageFormat::I420 => Err(ProcessingError::InvalidBuffer(
                "CpuImageProcessor does not support YUV destination".into(),
            )),
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
