use crate::device::Device;
use crate::error::InfersError;
use crate::tensor::TensorBuffer;
use infers_core::CpuImageBuffer;
#[cfg(target_os = "android")]
use platform_android::AndroidHardwareBufferHandle as CoreHardwareBuffer;
use processing::CpuImageProcessor;
#[cfg(feature = "vulkan")]
use processing::GpuImageProcessor;
use processing_core::{
    FitMode as CoreFitMode, ImageFormat as CoreImageFormat,
    ProcessingOptions as CoreProcessingOptions, Rotation as CoreRotation,
};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, uniffi::Enum)]
pub enum ImageFormat {
    Rgb888,
    Rgbf32,
    Nv12,
    I420,
}

uniffi_mirror! {
    ImageFormat <=> CoreImageFormat,
    [Rgb888, Rgbf32, Nv12, I420]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, uniffi::Enum)]
pub enum FitMode {
    Stretch,
    Contain,
    Crop,
}

uniffi_mirror! {
    FitMode <=> CoreFitMode,
    [Stretch, Contain, Crop]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, uniffi::Enum)]
pub enum Rotation {
    None,
    Rot90,
    Rot180,
    Rot270,
}

uniffi_mirror! {
    Rotation <=> CoreRotation,
    [None, Rot90, Rot180, Rot270]
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
    pub src_format: ImageFormat,
    pub dest_format: ImageFormat,
    pub fit_mode: FitMode,
    pub rotation: Rotation,
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
            src_format: opts.src_format.into(),
            dest_format: opts.dest_format.into(),
            fit_mode: opts.fit_mode.into(),
            rotation: opts.rotation.into(),
        }
    }
}

/// UniFFI-exported handle. Instantiable only on Android; off-Android the
/// `_unsupported: Infallible` field makes construction impossible so method
/// bodies that `match` on it are unreachable.
#[derive(uniffi::Object)]
pub struct HardwareBufferHandle {
    #[cfg(target_os = "android")]
    inner: Arc<CoreHardwareBuffer>,
    #[cfg(not(target_os = "android"))]
    _unsupported: std::convert::Infallible,
}

impl std::fmt::Debug for HardwareBufferHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        #[cfg(target_os = "android")]
        {
            f.debug_struct("HardwareBufferHandle")
                .field("width", &self.inner.width())
                .field("height", &self.inner.height())
                .finish()
        }
        #[cfg(not(target_os = "android"))]
        {
            let _ = f;
            match self._unsupported {}
        }
    }
}

#[cfg(target_os = "android")]
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
        #[cfg(target_os = "android")]
        {
            self.inner.width()
        }
        #[cfg(not(target_os = "android"))]
        {
            match self._unsupported {}
        }
    }

    pub fn height(&self) -> u32 {
        #[cfg(target_os = "android")]
        {
            self.inner.height()
        }
        #[cfg(not(target_os = "android"))]
        {
            match self._unsupported {}
        }
    }

    pub fn format(&self) -> u32 {
        #[cfg(target_os = "android")]
        {
            self.inner.desc().format
        }
        #[cfg(not(target_os = "android"))]
        {
            match self._unsupported {}
        }
    }

    pub fn raw_pointer(&self) -> u64 {
        #[cfg(target_os = "android")]
        {
            self.inner.raw_ptr() as u64
        }
        #[cfg(not(target_os = "android"))]
        {
            match self._unsupported {}
        }
    }

    pub fn lock_cpu(&self) -> Result<Vec<u8>, InfersError> {
        #[cfg(target_os = "android")]
        {
            self.inner.copy_cpu_packed().map_err(InfersError::from)
        }
        #[cfg(not(target_os = "android"))]
        {
            match self._unsupported {}
        }
    }
}

#[uniffi::export]
pub fn create_hardware_buffer_from_raw(
    ptr: u64,
    device: Device,
) -> Result<Arc<HardwareBufferHandle>, InfersError> {
    #[cfg(target_os = "android")]
    {
        let raw_ptr = ptr as *mut platform_android::ffi::AHardwareBuffer;
        // Acquire our own +1. Do not consume the caller's/Java object's ref —
        // `AHardwareBuffer_fromHardwareBuffer` is not reliable as a transferable
        // ownership handoff against `HardwareBuffer.close()` on MTE devices.
        let handle =
            CoreHardwareBuffer::from_raw(raw_ptr, device.into()).map_err(InfersError::from)?;
        Ok(Arc::new(HardwareBufferHandle::new(Arc::new(handle))))
    }
    #[cfg(not(target_os = "android"))]
    {
        let _ = (ptr, device);
        Err(InfersError::PlatformError {
            reason: "HardwareBuffer is only available on Android".into(),
        })
    }
}

#[derive(uniffi::Object)]
pub struct ImageProcessor {
    pub(crate) cpu_proc: Option<CpuImageProcessor>,
    #[cfg(feature = "vulkan")]
    pub(crate) gpu_proc: Option<GpuImageProcessor>,
    pub(crate) device: Device,
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

        #[cfg(feature = "vulkan")]
        if let Some(gpu) = &self.gpu_proc {
            let out_tensor = gpu.process(&cpu_img, &core_opts).map_err(InfersError::from)?;
            return Ok(Arc::new(TensorBuffer::from_boxed(out_tensor)));
        }

        if let Some(cpu) = &self.cpu_proc {
            let out_tensor = cpu.process(&cpu_img, &core_opts).map_err(InfersError::from)?;
            Ok(Arc::new(TensorBuffer::from_boxed(out_tensor)))
        } else {
            Err(InfersError::InternalError {
                reason: "No processor available".into(),
            })
        }
    }

    pub fn process_hardware_buffer(
        &self,
        buffer: Arc<HardwareBufferHandle>,
        options: ProcessingOptions,
    ) -> Result<Arc<TensorBuffer>, InfersError> {
        #[cfg(target_os = "android")]
        {
            let core_opts: CoreProcessingOptions = options.into();
            let hb = buffer.inner();

            #[cfg(feature = "vulkan")]
            if let Some(gpu) = &self.gpu_proc {
                let sampled = hb
                    .to_vulkan(Arc::clone(gpu.context()))
                    .map_err(InfersError::from)?;
                let out_tensor = gpu.process(&sampled, &core_opts).map_err(InfersError::from)?;
                return Ok(Arc::new(TensorBuffer::from_boxed(out_tensor)));
            }

            if let Some(cpu) = &self.cpu_proc {
                let data = hb.copy_cpu_packed().map_err(InfersError::from)?;
                let cpu_img = CpuImageBuffer::new(hb.width(), hb.height(), hb.format(), data)
                    .map_err(InfersError::from)?;
                let out_tensor = cpu.process(&cpu_img, &core_opts).map_err(InfersError::from)?;
                Ok(Arc::new(TensorBuffer::from_boxed(out_tensor)))
            } else {
                Err(InfersError::InternalError {
                    reason: "No processor available".into(),
                })
            }
        }
        #[cfg(not(target_os = "android"))]
        {
            let _ = (buffer, options);
            Err(InfersError::PlatformError {
                reason: "HardwareBuffer is only available on Android".into(),
            })
        }
    }
}

#[uniffi::export]
pub fn create_cpu_image_processor() -> Arc<ImageProcessor> {
    Arc::new(ImageProcessor {
        cpu_proc: Some(CpuImageProcessor::new()),
        #[cfg(feature = "vulkan")]
        gpu_proc: None,
        device: infers_core::Device::cpu().into(),
    })
}
