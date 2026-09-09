use crate::device::DeviceInfo;
use crate::error::InfersError;
use infers_core::CpuImage as CoreCpuImage;
use infers_core::HardwareImage as CoreHardwareImage;
use infers_core::Image;
use processing::CpuImageProcessor as CoreCpuImageProcessor;
use processing_core::{
    FitMode as CoreFitMode, ImageFormat as CoreImageFormat,
    ProcessingOptions as CoreProcessingOptions, TensorLayout as CoreTensorLayout,
};
use std::sync::Arc;

#[cfg(target_os = "android")]
use parking_lot::Mutex;

#[cfg(feature = "vulkan")]
use infers_gpu::VulkanImage;
#[cfg(feature = "vulkan")]
use processing::GpuImageProcessor as CoreGpuImageProcessor;

#[cfg(target_os = "android")]
use platform_android::AndroidHardwareBufferHandle as CoreHardwareBuffer;

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
pub enum TensorLayout {
    Nhwc,
    Nchw,
}

uniffi_mirror! {
    TensorLayout <=> CoreTensorLayout,
    [Nhwc, Nchw]
}

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
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
    pub rotation_degrees: f32,
    pub dest_layout: TensorLayout,
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
            rotation_degrees: opts.rotation_degrees,
            dest_layout: opts.dest_layout.into(),
        }
    }
}

/// Host-resident image bytes (camera frame, decoded file, etc.).
#[derive(uniffi::Object)]
pub struct HardwareImage {
    inner: CoreHardwareImage,
}

impl HardwareImage {
    pub(crate) fn inner(&self) -> &CoreHardwareImage {
        &self.inner
    }

    pub(crate) fn from_inner(inner: CoreHardwareImage) -> Self {
        Self { inner }
    }
}

#[uniffi::export]
impl HardwareImage {
    pub fn width(&self) -> u32 {
        self.inner.width()
    }

    pub fn height(&self) -> u32 {
        self.inner.height()
    }

    pub fn format(&self) -> ImageFormat {
        self.inner.format().into()
    }
}

#[uniffi::export]
pub fn create_hardware_image(
    width: u32,
    height: u32,
    format: ImageFormat,
    data: Vec<u8>,
) -> Result<Arc<HardwareImage>, InfersError> {
    let image = CoreHardwareImage::new(width, height, format.into(), data).map_err(InfersError::from)?;
    Ok(Arc::new(HardwareImage::from_inner(image)))
}

/// UniFFI-exported handle. Instantiable only on Android; off-Android the
/// `_unsupported: Infallible` field makes construction impossible so method
/// bodies that `match` on it are unreachable.
///
/// On Android the native AHB reference is uniquely owned inside a
/// `Mutex<Option<_>>` so `on()` can move it into Deferred without a second
/// acquire, and a later `close` / drop is a no-op once consumed.
#[derive(uniffi::Object)]
pub struct HardwareBufferHandle {
    #[cfg(target_os = "android")]
    inner: Mutex<Option<CoreHardwareBuffer>>,
    #[cfg(not(target_os = "android"))]
    _unsupported: std::convert::Infallible,
}

impl std::fmt::Debug for HardwareBufferHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        #[cfg(target_os = "android")]
        {
            let guard = self.inner.lock();
            match guard.as_ref() {
                Some(buf) => f
                    .debug_struct("HardwareBufferHandle")
                    .field("width", &buf.width())
                    .field("height", &buf.height())
                    .finish(),
                None => f
                    .debug_struct("HardwareBufferHandle")
                    .field("consumed", &true)
                    .finish(),
            }
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
    pub fn new(inner: CoreHardwareBuffer) -> Self {
        Self {
            inner: Mutex::new(Some(inner)),
        }
    }

    fn with_inner<R>(
        &self,
        f: impl FnOnce(&CoreHardwareBuffer) -> R,
    ) -> Result<R, InfersError> {
        let guard = self.inner.lock();
        let buf = guard.as_ref().ok_or(InfersError::AlreadyConsumed)?;
        Ok(f(buf))
    }

    fn take_inner(&self) -> Result<CoreHardwareBuffer, InfersError> {
        self.inner.lock().take().ok_or(InfersError::AlreadyConsumed)
    }
}

#[uniffi::export]
impl HardwareBufferHandle {
    pub fn width(&self) -> Result<u32, InfersError> {
        #[cfg(target_os = "android")]
        {
            self.with_inner(|buf| buf.width())
        }
        #[cfg(not(target_os = "android"))]
        {
            match self._unsupported {}
        }
    }

    pub fn height(&self) -> Result<u32, InfersError> {
        #[cfg(target_os = "android")]
        {
            self.with_inner(|buf| buf.height())
        }
        #[cfg(not(target_os = "android"))]
        {
            match self._unsupported {}
        }
    }

    pub fn format(&self) -> Result<u32, InfersError> {
        #[cfg(target_os = "android")]
        {
            self.with_inner(|buf| buf.desc().format)
        }
        #[cfg(not(target_os = "android"))]
        {
            match self._unsupported {}
        }
    }

    pub fn raw_pointer(&self) -> Result<u64, InfersError> {
        #[cfg(target_os = "android")]
        {
            self.with_inner(|buf| buf.raw_ptr() as u64)
        }
        #[cfg(not(target_os = "android"))]
        {
            match self._unsupported {}
        }
    }

    pub fn lock_cpu(&self) -> Result<Vec<u8>, InfersError> {
        #[cfg(target_os = "android")]
        {
            self.with_inner(|buf| buf.copy_cpu_packed())?
                .map_err(InfersError::from)
        }
        #[cfg(not(target_os = "android"))]
        {
            match self._unsupported {}
        }
    }
}

/// Defer zero-copy Vulkan import of this buffer until Pending materialize.
///
/// Consumes the owned AHB reference; subsequent use of this handle fails with
/// [`InfersError::AlreadyConsumed`].
#[cfg(feature = "vulkan")]
#[uniffi::export]
impl HardwareBufferHandle {
    pub fn on(
        self: Arc<Self>,
        device: Arc<crate::gpu_device::GpuDevice>,
    ) -> Result<Arc<crate::deferred::GpuDeferred>, InfersError> {
        #[cfg(target_os = "android")]
        {
            let owned = self.take_inner()?;
            Ok(Arc::new(crate::deferred::GpuDeferred::from_inner(
                owned.on(device.vulkan()),
            )))
        }
        #[cfg(not(target_os = "android"))]
        {
            let _ = (self, device);
            Err(InfersError::PlatformError {
                reason: "HardwareBuffer is only available on Android".into(),
            })
        }
    }
}

/// Wrap a borrowed `AHardwareBuffer*` from `AHardwareBuffer_fromHardwareBuffer`.
///
/// Acquires an independent +1 so Rust owns a reference separate from the Java
/// `HardwareBuffer`. Does not release the Java object's reference.
#[uniffi::export]
pub fn create_hardware_buffer_from_java(
    ptr: u64,
    device: DeviceInfo,
) -> Result<Arc<HardwareBufferHandle>, InfersError> {
    #[cfg(target_os = "android")]
    {
        let raw_ptr = ptr as *mut platform_android::ffi::AHardwareBuffer;
        let info: infers_core::DeviceInfo = device.into();
        let handle =
            CoreHardwareBuffer::from_borrowed(raw_ptr, info).map_err(InfersError::from)?;
        Ok(Arc::new(HardwareBufferHandle::new(handle)))
    }
    #[cfg(not(target_os = "android"))]
    {
        let _ = (ptr, device);
        Err(InfersError::PlatformError {
            reason: "HardwareBuffer is only available on Android".into(),
        })
    }
}

/// Take ownership of an `AHardwareBuffer*` that already carries a +1 reference
/// (for example from `AHardwareBuffer_allocate`). Does not acquire.
#[uniffi::export]
pub fn create_hardware_buffer_from_owned(
    ptr: u64,
    device: DeviceInfo,
) -> Result<Arc<HardwareBufferHandle>, InfersError> {
    #[cfg(target_os = "android")]
    {
        let raw_ptr = ptr as *mut platform_android::ffi::AHardwareBuffer;
        let info: infers_core::DeviceInfo = device.into();
        let handle =
            CoreHardwareBuffer::from_owned(raw_ptr, info).map_err(InfersError::from)?;
        Ok(Arc::new(HardwareBufferHandle::new(handle)))
    }
    #[cfg(not(target_os = "android"))]
    {
        let _ = (ptr, device);
        Err(InfersError::PlatformError {
            reason: "HardwareBuffer is only available on Android".into(),
        })
    }
}

/// CPU-resident image (zero-copy wrap of [`HardwareImage`] bytes).
#[derive(uniffi::Object)]
pub struct CpuImage {
    inner: CoreCpuImage,
}

impl CpuImage {
    pub(crate) fn from_core(inner: CoreCpuImage) -> Self {
        Self { inner }
    }

    #[allow(dead_code)]
    pub(crate) fn inner(&self) -> &CoreCpuImage {
        &self.inner
    }
}

#[uniffi::export]
impl CpuImage {
    pub fn width(&self) -> u32 {
        self.inner.width()
    }

    pub fn height(&self) -> u32 {
        self.inner.height()
    }

    pub fn format(&self) -> ImageFormat {
        self.inner.format().into()
    }
}

#[derive(uniffi::Object)]
pub struct CpuImageProcessor {
    inner: CoreCpuImageProcessor,
}

impl CpuImageProcessor {
    pub(crate) fn core(&self) -> &CoreCpuImageProcessor {
        &self.inner
    }
}

#[uniffi::export]
impl CpuImageProcessor {
    #[uniffi::constructor]
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: CoreCpuImageProcessor::new(),
        })
    }

    pub fn device_info(&self) -> DeviceInfo {
        infers_core::Cpu::info().clone().into()
    }
}

#[cfg(feature = "vulkan")]
#[derive(uniffi::Object)]
pub struct GpuImage {
    inner: VulkanImage,
}

#[cfg(feature = "vulkan")]
impl GpuImage {
    pub(crate) fn from_vulkan(inner: VulkanImage) -> Self {
        Self { inner }
    }

    #[allow(dead_code)]
    pub(crate) fn inner(&self) -> &VulkanImage {
        &self.inner
    }
}

#[cfg(feature = "vulkan")]
#[uniffi::export]
impl GpuImage {
    pub fn width(&self) -> u32 {
        self.inner.width()
    }

    pub fn height(&self) -> u32 {
        self.inner.height()
    }

    pub fn format(&self) -> ImageFormat {
        self.inner.format().into()
    }
}

#[cfg(feature = "vulkan")]
#[derive(uniffi::Object)]
pub struct GpuImageProcessor {
    pub(crate) inner: CoreGpuImageProcessor,
}

impl GpuImageProcessor {
    pub(crate) fn core(&self) -> &CoreGpuImageProcessor {
        &self.inner
    }
}

#[cfg(feature = "vulkan")]
#[uniffi::export]
impl GpuImageProcessor {
    pub fn device_info(&self) -> DeviceInfo {
        self.inner.vulkan().info().clone().into()
    }
}
