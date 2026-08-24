use crate::device::Device;
use crate::error::InfersError;
use crate::tensor::TensorBuffer;
use infers_core::{CpuImageBuffer, ImageInputBuffer};
use platform_android::AndroidHardwareBufferHandle as CoreHardwareBuffer;
use processing::{CpuImageProcessor, GpuImageProcessor};
use processing_core::{
    FitMode as CoreFitMode, ImageFormat as CoreImageFormat,
    ProcessingOptions as CoreProcessingOptions, Rotation as CoreRotation,
};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, uniffi::Enum)]
pub enum ImageFormat {
    Rgb888,
    Rgbf32,
}

impl From<CoreImageFormat> for ImageFormat {
    fn from(format: CoreImageFormat) -> Self {
        match format {
            CoreImageFormat::RGB888 => ImageFormat::Rgb888,
            CoreImageFormat::RGBF32 => ImageFormat::Rgbf32,
        }
    }
}

impl From<ImageFormat> for CoreImageFormat {
    fn from(format: ImageFormat) -> Self {
        match format {
            ImageFormat::Rgb888 => CoreImageFormat::RGB888,
            ImageFormat::Rgbf32 => CoreImageFormat::RGBF32,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, uniffi::Enum)]
pub enum FitMode {
    Stretch,
    Contain,
    Crop,
}

impl From<CoreFitMode> for FitMode {
    fn from(mode: CoreFitMode) -> Self {
        match mode {
            CoreFitMode::STRETCH => FitMode::Stretch,
            CoreFitMode::CONTAIN => FitMode::Contain,
            CoreFitMode::CROP => FitMode::Crop,
        }
    }
}

impl From<FitMode> for CoreFitMode {
    fn from(mode: FitMode) -> Self {
        match mode {
            FitMode::Stretch => CoreFitMode::STRETCH,
            FitMode::Contain => CoreFitMode::CONTAIN,
            FitMode::Crop => CoreFitMode::CROP,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, uniffi::Enum)]
pub enum Rotation {
    None,
    R90deg,
    R180deg,
    R270deg,
}

impl From<CoreRotation> for Rotation {
    fn from(rot: CoreRotation) -> Self {
        match rot {
            CoreRotation::None => Rotation::None,
            CoreRotation::R90DEG => Rotation::R90deg,
            CoreRotation::R180DEG => Rotation::R180deg,
            CoreRotation::R270DEG => Rotation::R270deg,
        }
    }
}

impl From<Rotation> for CoreRotation {
    fn from(rot: Rotation) -> Self {
        match rot {
            Rotation::None => CoreRotation::None,
            Rotation::R90deg => CoreRotation::R90DEG,
            Rotation::R180deg => CoreRotation::R180DEG,
            Rotation::R270deg => CoreRotation::R270DEG,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, uniffi::Record)]
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

impl From<CoreProcessingOptions> for ProcessingOptions {
    fn from(opts: CoreProcessingOptions) -> Self {
        Self {
            src_w: opts.src_w,
            src_h: opts.src_h,
            crop_x: opts.crop_x,
            crop_y: opts.crop_y,
            crop_w: opts.crop_w,
            crop_h: opts.crop_h,
            dest_w: opts.dest_w,
            dest_h: opts.dest_h,
            dest_format: opts.dest_format.into(),
            fit_mode: opts.fit_mode.into(),
            rotation: opts.rotation.into(),
        }
    }
}

impl From<ProcessingOptions> for CoreProcessingOptions {
    fn from(opts: ProcessingOptions) -> Self {
        Self {
            src_w: opts.src_w,
            src_h: opts.src_h,
            crop_x: opts.crop_x,
            crop_y: opts.crop_y,
            crop_w: opts.crop_w,
            crop_h: opts.crop_h,
            dest_w: opts.dest_w,
            dest_h: opts.dest_h,
            dest_format: opts.dest_format.into(),
            fit_mode: opts.fit_mode.into(),
            rotation: opts.rotation.into(),
        }
    }
}

#[derive(uniffi::Object)]
pub struct HardwareBufferHandle {
    inner: Arc<CoreHardwareBuffer>,
}

impl std::fmt::Debug for HardwareBufferHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HardwareBufferHandle")
            .field("width", &self.inner.width())
            .field("height", &self.inner.height())
            .finish()
    }
}

impl HardwareBufferHandle {
    pub fn new(inner: Arc<CoreHardwareBuffer>) -> Self {
        Self { inner }
    }

    pub fn inner(&self) -> &CoreHardwareBuffer {
        &self.inner
    }
}

#[uniffi::export]
impl HardwareBufferHandle {
    pub fn width(&self) -> u32 {
        self.inner.width()
    }

    pub fn height(&self) -> u32 {
        self.inner.height()
    }

    pub fn format(&self) -> u32 {
        self.inner.desc().format
    }

    pub fn raw_pointer(&self) -> u64 {
        self.inner.raw_ptr() as u64
    }

    pub fn lock_cpu(&self) -> Result<Vec<u8>, InfersError> {
        let locked = self.inner.lock_cpu_read().map_err(InfersError::from)?;
        Ok(locked.as_slice().to_vec())
    }
}

#[uniffi::export]
pub fn create_hardware_buffer_from_raw(
    ptr: u64,
    device: Device,
) -> Result<Arc<HardwareBufferHandle>, InfersError> {
    let raw_ptr = ptr as *mut platform_android::ffi::AHardwareBuffer;
    let handle =
        CoreHardwareBuffer::from_raw(raw_ptr, device.into()).map_err(InfersError::from)?;
    Ok(Arc::new(HardwareBufferHandle::new(Arc::new(handle))))
}

#[derive(uniffi::Object)]
pub struct ImageProcessor {
    cpu_proc: Option<CpuImageProcessor>,
    gpu_proc: Option<GpuImageProcessor>,
    device: Device,
}

impl std::fmt::Debug for ImageProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImageProcessor")
            .field("device", &self.device)
            .finish()
    }
}

#[uniffi::export]
impl ImageProcessor {
    pub fn device(&self) -> Device {
        self.device.clone()
    }

    pub fn process_bytes(
        &self,
        data: Vec<u8>,
        width: u32,
        height: u32,
        format: ImageFormat,
        options: ProcessingOptions,
    ) -> Result<Arc<TensorBuffer>, InfersError> {
        let cpu_img = CpuImageBuffer::new(width, height, format.into(), data)
            .map_err(InfersError::from)?;
        let core_opts: CoreProcessingOptions = options.into();

        if let Some(gpu) = &self.gpu_proc {
            let out_tensor = gpu.process(&cpu_img, &core_opts).map_err(InfersError::from)?;
            Ok(Arc::new(TensorBuffer::from_boxed(out_tensor)))
        } else if let Some(cpu) = &self.cpu_proc {
            let out_tensor = cpu.process(&cpu_img, &core_opts).map_err(InfersError::from)?;
            Ok(Arc::new(TensorBuffer::from_boxed(out_tensor)))
        } else {
            Err(InfersError::InternalError {
                message: "No processor available".into(),
            })
        }
    }

    pub fn process_hardware_buffer(
        &self,
        buffer: Arc<HardwareBufferHandle>,
        options: ProcessingOptions,
    ) -> Result<Arc<TensorBuffer>, InfersError> {
        let core_opts: CoreProcessingOptions = options.into();
        let hb = buffer.inner();

        if let Some(gpu) = &self.gpu_proc {
            let out_tensor = gpu.process(hb, &core_opts).map_err(InfersError::from)?;
            Ok(Arc::new(TensorBuffer::from_boxed(out_tensor)))
        } else if let Some(cpu) = &self.cpu_proc {
            let out_tensor = cpu.process(hb, &core_opts).map_err(InfersError::from)?;
            Ok(Arc::new(TensorBuffer::from_boxed(out_tensor)))
        } else {
            Err(InfersError::InternalError {
                message: "No processor available".into(),
            })
        }
    }
}

#[uniffi::export]
pub fn create_cpu_image_processor() -> Arc<ImageProcessor> {
    Arc::new(ImageProcessor {
        cpu_proc: Some(CpuImageProcessor::new()),
        gpu_proc: None,
        device: infers_core::Device::cpu().into(),
    })
}

#[uniffi::export]
pub fn create_gpu_image_processor(device: Device) -> Result<Arc<ImageProcessor>, InfersError> {
    let core_device: infers_core::Device = device.clone().into();
    let gpu_proc = GpuImageProcessor::new(&core_device).map_err(InfersError::from)?;
    Ok(Arc::new(ImageProcessor {
        cpu_proc: None,
        gpu_proc: Some(gpu_proc),
        device,
    }))
}
