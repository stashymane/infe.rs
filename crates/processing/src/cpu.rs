use infers_core::{
    CoreError, Cpu, HardwareImage, HostBytes, ImageFormat, MaterializeTarget, ProcessingOptions,
    Tensor, TensorShape,
};
use parking_lot::Mutex;
use processing_core::{
    FitMode, TensorLayout, convert_storage, dest_buffer_bytes,
};
use std::sync::Arc;

/// High-performance CPU image processor using the shared dest-centric convert kernel.
#[derive(Clone)]
pub struct CpuImageProcessor {
    pool: Arc<Mutex<DestPool>>,
}

struct DestPool {
    slots: [Vec<u8>; 2],
    next: usize,
}

impl DestPool {
    fn acquire(&mut self, needed: usize) -> Vec<u8> {
        let i = self.next & 1;
        self.next ^= 1;
        let mut buf = std::mem::replace(&mut self.slots[i], Vec::with_capacity(needed));
        buf.clear();
        buf.resize(needed, 0);
        buf
    }
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
            pool: Arc::new(Mutex::new(DestPool {
                slots: [Vec::new(), Vec::new()],
                next: 0,
            })),
        }
    }

    pub(crate) fn materialize_from_hardware(
        &self,
        input: Arc<HardwareImage>,
        options: &ProcessingOptions,
        _target: MaterializeTarget,
    ) -> Result<Tensor<Cpu>, CoreError> {
        self.process_hardware(&input, options)
    }

    fn process_hardware(
        &self,
        input: &HardwareImage,
        options: &ProcessingOptions,
    ) -> Result<Tensor<Cpu>, CoreError> {
        let mut params = *options;
        params.src_format = input.format();
        params.src_w = input.width();
        params.src_h = input.height();

        validate_params(input, &params)?;

        let (shape, dtype) = output_shape_dtype(&params)?;
        let needed = dest_buffer_bytes(&params) as usize;
        if needed != shape.byte_size(dtype) {
            return Err(CoreError::InvalidImageBuffer(format!(
                "Dest buffer size {needed} does not match tensor byte size {}",
                shape.byte_size(dtype)
            )));
        }

        let mut dst = self.pool.lock().acquire(needed);
        let src = input.as_bytes();

        if can_identity_memcpy(&params) {
            identity_memcpy_rgb888_nhwc(src, &mut dst, &params)?;
        } else {
            convert_storage(src, &mut dst, &params);
        }

        Ok(Tensor::from_storage(Cpu, shape, dtype, HostBytes::from_vec(dst)))
    }
}

fn validate_params(input: &HardwareImage, params: &ProcessingOptions) -> Result<(), CoreError> {
    match params.src_format {
        ImageFormat::Rgb888 | ImageFormat::Nv12 | ImageFormat::I420 => {}
        ImageFormat::Rgbf32 => {
            return Err(CoreError::InvalidImageBuffer(
                "CpuImageProcessor does not support RGBF32 source".into(),
            ));
        }
    }

    if params.dest_format.is_yuv() {
        return Err(CoreError::InvalidImageBuffer(
            "CpuImageProcessor does not support YUV destination".into(),
        ));
    }

    if params.src_format.is_yuv()
        && (!params.src_w.is_multiple_of(2) || !params.src_h.is_multiple_of(2))
    {
        return Err(CoreError::InvalidImageBuffer(
            "YUV 4:2:0 input width and height must be even".into(),
        ));
    }

    let expected = params.src_format.frame_bytes(params.src_w, params.src_h) as usize;
    if input.as_bytes().len() < expected {
        return Err(CoreError::InvalidImageBuffer(format!(
            "Input buffer holds {} bytes but {}x{} {:?} needs {}",
            input.as_bytes().len(),
            params.src_w,
            params.src_h,
            params.src_format,
            expected
        )));
    }

    let (crop_x, crop_y, crop_w, crop_h) = params.effective_crop();
    if crop_w == 0 || crop_h == 0 {
        return Err(CoreError::InvalidImageBuffer(
            "Effective crop width and height must be greater than zero".into(),
        ));
    }

    let exceeds = crop_x.checked_add(crop_w).is_none_or(|r| r > params.src_w)
        || crop_y.checked_add(crop_h).is_none_or(|b| b > params.src_h);
    if exceeds {
        return Err(CoreError::InvalidImageBuffer(format!(
            "Crop region ({}, {}, {}, {}) exceeds source image dimensions ({}x{})",
            crop_x, crop_y, crop_w, crop_h, params.src_w, params.src_h
        )));
    }

    if params.dest_w == 0 || params.dest_h == 0 {
        return Err(CoreError::InvalidImageBuffer(
            "Destination dimensions must be non-zero".into(),
        ));
    }

    Ok(())
}

fn can_identity_memcpy(params: &ProcessingOptions) -> bool {
    if params.fit_mode != FitMode::Stretch
        || params.rotation_degrees != 0.0
        || params.src_format != ImageFormat::Rgb888
        || params.dest_format != ImageFormat::Rgb888
        || params.dest_layout != TensorLayout::Nhwc
    {
        return false;
    }
    let (_x, _y, crop_w, crop_h) = params.effective_crop();
    crop_w == params.dest_w && crop_h == params.dest_h
}

fn identity_memcpy_rgb888_nhwc(
    src: &[u8],
    dst: &mut [u8],
    params: &ProcessingOptions,
) -> Result<(), CoreError> {
    let (crop_x, crop_y, crop_w, crop_h) = params.effective_crop();
    let src_w = params.src_w as usize;
    let row_bytes = (crop_w as usize) * 3;
    for y in 0..crop_h as usize {
        let src_off = ((crop_y as usize + y) * src_w + crop_x as usize) * 3;
        let dst_off = y * row_bytes;
        let src_end = src_off + row_bytes;
        let dst_end = dst_off + row_bytes;
        if src_end > src.len() || dst_end > dst.len() {
            return Err(CoreError::InvalidImageBuffer(
                "Identity memcpy region exceeds buffer bounds".into(),
            ));
        }
        dst[dst_off..dst_end].copy_from_slice(&src[src_off..src_end]);
    }
    Ok(())
}

pub(crate) fn output_shape_dtype(
    options: &ProcessingOptions,
) -> Result<(TensorShape, infers_core::DataType), CoreError> {
    let (dest_w, dest_h) = (options.dest_w, options.dest_h);
    let shape = match options.dest_layout {
        TensorLayout::Nhwc => TensorShape::new(vec![1, dest_h as usize, dest_w as usize, 3])?,
        TensorLayout::Nchw => TensorShape::new(vec![1, 3, dest_h as usize, dest_w as usize])?,
    };
    let dtype = match options.dest_format {
        ImageFormat::Rgb888 => infers_core::DataType::U8,
        ImageFormat::Rgbf32 => infers_core::DataType::F32,
        ImageFormat::Nv12 | ImageFormat::I420 => {
            return Err(CoreError::InvalidImageBuffer(
                "CPU convert kernels emit RGB888 or RGBF32, not YUV".into(),
            ));
        }
    };
    Ok((shape, dtype))
}
