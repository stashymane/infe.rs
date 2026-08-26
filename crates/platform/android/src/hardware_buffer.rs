use crate::error::AndroidPlatformError;
use crate::ffi::*;
use infers_core::{Device, ImageFormat, ImageInputBuffer};
use std::ffi::c_void;

/// Safe wrapper around an Android `AHardwareBuffer`
pub struct AndroidHardwareBufferHandle {
    raw_ptr: *mut AHardwareBuffer,
    desc: AHardwareBuffer_Desc,
    format: ImageFormat,
    device: Device,
}

// SAFETY: `AHardwareBuffer` is reference-counted by the platform and its
// `acquire`/`release`/`describe` entry points are documented as thread-safe. The
// only interior mutation this wrapper performs is through `lock_cpu_read`, which
// takes `&self` and hands back a lock guard borrowing `self`; concurrent locks
// are permitted by the NDK as long as every lock is paired with an unlock, which
// `LockedCpuBuffer` guarantees.
unsafe impl Send for AndroidHardwareBufferHandle {}
unsafe impl Sync for AndroidHardwareBufferHandle {}

impl AndroidHardwareBufferHandle {
    /// Wrap an existing `AHardwareBuffer` pointer, incrementing its reference count.
    ///
    /// # Safety-relevant preconditions
    ///
    /// `raw_ptr` must be either null or a currently-live `AHardwareBuffer`
    /// obtained from the platform. A dangling or foreign pointer cannot be
    /// detected here and results in undefined behaviour.
    pub fn from_raw(
        raw_ptr: *mut AHardwareBuffer,
        device: Device,
    ) -> Result<Self, AndroidPlatformError> {
        if raw_ptr.is_null() {
            return Err(AndroidPlatformError::NullBufferPointer);
        }

        // SAFETY: `raw_ptr` is non-null and the caller guarantees it refers to a
        // live buffer, so acquiring adds a reference we own from here on.
        unsafe { AHardwareBuffer_acquire(raw_ptr) };

        // SAFETY: we just took a reference that this call takes ownership of.
        match unsafe { Self::from_acquired(raw_ptr, device) } {
            Ok(handle) => Ok(handle),
            Err(err) => {
                // Release the reference acquired above so a rejected buffer does
                // not leak.
                // SAFETY: the acquire above succeeded and no handle owns it.
                unsafe { AHardwareBuffer_release(raw_ptr) };
                Err(err)
            }
        }
    }

    /// Take ownership of a pointer returned by `AHardwareBuffer_allocate` (no extra acquire).
    ///
    /// On error the buffer is released, since this call takes ownership of the
    /// caller's reference.
    pub fn from_allocated(
        raw_ptr: *mut AHardwareBuffer,
        device: Device,
    ) -> Result<Self, AndroidPlatformError> {
        if raw_ptr.is_null() {
            return Err(AndroidPlatformError::NullBufferPointer);
        }

        // SAFETY: caller transfers their reference to us.
        match unsafe { Self::from_acquired(raw_ptr, device) } {
            Ok(handle) => Ok(handle),
            Err(err) => {
                // SAFETY: we own the caller's reference and no handle was built.
                unsafe { AHardwareBuffer_release(raw_ptr) };
                Err(err)
            }
        }
    }

    /// # Safety
    ///
    /// Takes ownership of one reference to `raw_ptr`, which must be non-null and
    /// point to a live `AHardwareBuffer`. On success the returned handle releases
    /// that reference on drop; on error the caller must release it.
    unsafe fn from_acquired(
        raw_ptr: *mut AHardwareBuffer,
        device: Device,
    ) -> Result<Self, AndroidPlatformError> {
        // SAFETY: `AHardwareBuffer_Desc` is a plain `#[repr(C)]` struct of
        // integers, so an all-zero value is a valid initial state for the
        // platform to overwrite.
        let mut desc: AHardwareBuffer_Desc = unsafe { std::mem::zeroed() };
        // SAFETY: `raw_ptr` is live per this function's contract and `desc` is a
        // valid, uniquely-borrowed out-parameter.
        unsafe { AHardwareBuffer_describe(raw_ptr, &mut desc) };

        // Resolve the pixel format up front so that the handle can never
        // describe itself with a format that does not match its memory layout.
        let format = image_format_for(desc.format)?;

        Ok(Self {
            raw_ptr,
            desc,
            format,
            device,
        })
    }

    #[inline]
    pub fn raw_ptr(&self) -> *mut AHardwareBuffer {
        self.raw_ptr
    }

    #[inline]
    pub fn desc(&self) -> &AHardwareBuffer_Desc {
        &self.desc
    }

    #[inline]
    pub fn width(&self) -> u32 {
        self.desc.width
    }

    #[inline]
    pub fn height(&self) -> u32 {
        self.desc.height
    }

    /// The pixel format, validated when the handle was created.
    #[inline]
    pub fn format(&self) -> ImageFormat {
        self.format
    }

    #[inline]
    pub fn device(&self) -> &Device {
        &self.device
    }

    #[inline]
    pub fn supports_gpu_sampling(&self) -> bool {
        (self.desc.usage & AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE) != 0
    }

    /// Lock the buffer for CPU reading (RAII unlock on drop).
    ///
    /// Only formats with a single densely-packed plane can be mapped this way.
    /// Planar YUV buffers require `AHardwareBuffer_lockPlanes` and are rejected;
    /// use [`AndroidHardwareBufferHandle::to_vulkan`] and the GPU processor for
    /// those instead.
    pub fn lock_cpu_read(&self) -> Result<LockedCpuBuffer<'_>, AndroidPlatformError> {
        let len = self.mapped_len()?;

        let mut virtual_addr: *mut c_void = std::ptr::null_mut();
        // SAFETY: `raw_ptr` is live for `&self`, the usage flag is a valid
        // CPU-read flag, -1 means "no fence to wait on", a null rect requests
        // the whole buffer, and `virtual_addr` is a valid out-parameter.
        let status = unsafe {
            AHardwareBuffer_lock(
                self.raw_ptr,
                AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN,
                -1,
                std::ptr::null(),
                &mut virtual_addr,
            )
        };
        if status != 0 || virtual_addr.is_null() {
            return Err(AndroidPlatformError::LockFailed(status));
        }

        // SAFETY: the lock succeeded, so `virtual_addr` points to a readable
        // mapping of this buffer. `mapped_len` derived `len` from the buffer's
        // own stride, height and validated format, so it does not exceed the
        // mapping. The mapping stays valid until `AHardwareBuffer_unlock`, which
        // only happens when the returned guard is dropped or unlocked, and the
        // guard borrows `self` so the buffer outlives it.
        let slice = unsafe { std::slice::from_raw_parts(virtual_addr as *const u8, len) };

        Ok(LockedCpuBuffer {
            buffer: self,
            slice,
        })
    }

    /// Copy pixels into a tightly packed buffer (`width * bpp` bytes per row).
    ///
    /// `AHardwareBuffer` row pitch is `stride * bpp` and may exceed `width`; the
    /// CPU image processor expects unpadded rows.
    pub fn copy_cpu_packed(&self) -> Result<Vec<u8>, AndroidPlatformError> {
        let locked = self.lock_cpu_read()?;
        let bpp = match self.format {
            ImageFormat::Rgb888 => 3usize,
            ImageFormat::Rgbf32 => 12usize,
            ImageFormat::Nv12 | ImageFormat::I420 => {
                return Err(AndroidPlatformError::UnsupportedFormat(self.desc.format));
            }
        };

        let width = self.desc.width as usize;
        let height = self.desc.height as usize;
        let stride = self.desc.stride as usize;
        let src_pitch = stride
            .checked_mul(bpp)
            .ok_or(AndroidPlatformError::UnsupportedFormat(self.desc.format))?;
        let dst_pitch = width
            .checked_mul(bpp)
            .ok_or(AndroidPlatformError::UnsupportedFormat(self.desc.format))?;

        if src_pitch < dst_pitch {
            return Err(AndroidPlatformError::UnsupportedFormat(self.desc.format));
        }

        let src = locked.as_slice();
        if src_pitch == dst_pitch {
            let need = dst_pitch
                .checked_mul(height)
                .ok_or(AndroidPlatformError::UnsupportedFormat(self.desc.format))?;
            if src.len() < need {
                return Err(AndroidPlatformError::UnsupportedFormat(self.desc.format));
            }
            return Ok(src[..need].to_vec());
        }

        let mut out = Vec::with_capacity(
            dst_pitch
                .checked_mul(height)
                .ok_or(AndroidPlatformError::UnsupportedFormat(self.desc.format))?,
        );
        for y in 0..height {
            let start = y
                .checked_mul(src_pitch)
                .ok_or(AndroidPlatformError::UnsupportedFormat(self.desc.format))?;
            let end = start
                .checked_add(dst_pitch)
                .ok_or(AndroidPlatformError::UnsupportedFormat(self.desc.format))?;
            if end > src.len() {
                return Err(AndroidPlatformError::UnsupportedFormat(self.desc.format));
            }
            out.extend_from_slice(&src[start..end]);
        }
        Ok(out)
    }

    /// Byte length of the CPU mapping produced by `AHardwareBuffer_lock`.
    ///
    /// `desc.stride` is measured in pixels, so the row pitch is
    /// `stride * bytes_per_pixel`. Computed in `u64` so that a hostile or
    /// corrupt descriptor cannot wrap around into a short length.
    fn mapped_len(&self) -> Result<usize, AndroidPlatformError> {
        let bytes_per_pixel = match self.format {
            ImageFormat::Rgb888 => 3u64,
            ImageFormat::Rgbf32 => 12u64,
            // Planar formats have per-plane strides that a single `lock` call
            // does not describe; refuse rather than guess at the layout.
            ImageFormat::Nv12 | ImageFormat::I420 => {
                return Err(AndroidPlatformError::UnsupportedFormat(self.desc.format));
            }
        };

        let len = u64::from(self.desc.stride)
            .checked_mul(bytes_per_pixel)
            .and_then(|row| row.checked_mul(u64::from(self.desc.height)))
            .ok_or(AndroidPlatformError::UnsupportedFormat(self.desc.format))?;

        usize::try_from(len).map_err(|_| AndroidPlatformError::UnsupportedFormat(self.desc.format))
    }
}

/// Map an `AHARDWAREBUFFER_FORMAT_*` value onto the pixel format the processing
/// pipeline understands.
///
/// Formats whose byte layout has no equivalent in [`ImageFormat`] are rejected
/// rather than approximated, because a mismatch between the reported format and
/// the real layout causes both garbled output and out-of-bounds reads.
#[allow(non_upper_case_globals)]
fn image_format_for(ahb_format: u32) -> Result<ImageFormat, AndroidPlatformError> {
    match ahb_format {
        AHARDWAREBUFFER_FORMAT_R8G8B8_UNORM => Ok(ImageFormat::Rgb888),
        AHARDWAREBUFFER_FORMAT_Y8Cb8Cr8_420 => Ok(ImageFormat::Nv12),
        _ => Err(AndroidPlatformError::UnsupportedFormat(ahb_format)),
    }
}

impl Drop for AndroidHardwareBufferHandle {
    fn drop(&mut self) {
        // SAFETY: the handle owns one reference to a live buffer, established at
        // construction and released exactly once here.
        unsafe { AHardwareBuffer_release(self.raw_ptr) };
    }
}

impl ImageInputBuffer for AndroidHardwareBufferHandle {
    fn width(&self) -> u32 {
        self.desc.width
    }

    fn height(&self) -> u32 {
        self.desc.height
    }

    fn format(&self) -> ImageFormat {
        self.format
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn as_bytes(&self) -> Option<&[u8]> {
        None
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// RAII lock for CPU memory reading from an `AHardwareBuffer`
pub struct LockedCpuBuffer<'a> {
    buffer: &'a AndroidHardwareBufferHandle,
    slice: &'a [u8],
}

impl LockedCpuBuffer<'_> {
    pub fn as_slice(&self) -> &[u8] {
        self.slice
    }

    /// Unlock explicitly, reporting a platform failure that `Drop` would discard.
    pub fn unlock(self) -> Result<(), AndroidPlatformError> {
        let status = self.unlock_raw();
        // Skip the `Drop` impl so the buffer is not unlocked twice.
        std::mem::forget(self);
        if status == 0 {
            Ok(())
        } else {
            Err(AndroidPlatformError::UnlockFailed(status))
        }
    }

    fn unlock_raw(&self) -> i32 {
        let mut fence: i32 = -1;
        // SAFETY: this guard exists only while the buffer is locked, so exactly
        // one unlock is owed; `fence` is a valid out-parameter.
        unsafe { AHardwareBuffer_unlock(self.buffer.raw_ptr, &mut fence) }
    }
}

impl Drop for LockedCpuBuffer<'_> {
    fn drop(&mut self) {
        // A failure here cannot be propagated out of `Drop`; callers that need to
        // observe it should use `unlock` instead.
        let _ = self.unlock_raw();
    }
}
