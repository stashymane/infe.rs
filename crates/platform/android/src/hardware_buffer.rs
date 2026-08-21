use crate::error::AndroidPlatformError;
use crate::ffi::*;
use crate::vulkan::VulkanHardwareBufferImport;
use infers_core::{Device, ImageFormat, ImageInputBuffer};
use std::ffi::c_void;

/// Safe wrapper around an Android `AHardwareBuffer`
pub struct AndroidHardwareBufferHandle {
    raw_ptr: *mut AHardwareBuffer,
    desc: AHardwareBuffer_Desc,
    device: Device,
    host_mock_data: Option<Vec<u8>>,
}

// Safety: AHardwareBuffer instances are ref-counted and thread-safe across threads on Android
unsafe impl Send for AndroidHardwareBufferHandle {}
unsafe impl Sync for AndroidHardwareBufferHandle {}

impl AndroidHardwareBufferHandle {
    /// Create a handle wrapping an existing raw `AHardwareBuffer` pointer
    pub fn from_raw(raw_ptr: *mut AHardwareBuffer, device: Device) -> Result<Self, AndroidPlatformError> {
        if raw_ptr.is_null() {
            return Err(AndroidPlatformError::NullBufferPointer);
        }

        #[cfg(target_os = "android")]
        unsafe {
            AHardwareBuffer_acquire(raw_ptr);
            let mut desc = std::mem::zeroed();
            AHardwareBuffer_describe(raw_ptr, &mut desc);
            Ok(Self {
                raw_ptr,
                desc,
                device,
                host_mock_data: None,
            })
        }

        #[cfg(not(target_os = "android"))]
        {
            let desc = AHardwareBuffer_Desc {
                width: 0,
                height: 0,
                layers: 1,
                format: AHARDWAREBUFFER_FORMAT_R8G8B8_UNORM,
                usage: AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE,
                stride: 0,
                rfu0: 0,
                rfu1: 0,
            };
            Ok(Self {
                raw_ptr,
                desc,
                device,
                host_mock_data: None,
            })
        }
    }

    /// Create a simulated hardware buffer handle for testing on host/desktop
    pub fn from_mock(
        width: u32,
        height: u32,
        format: ImageFormat,
        data: Vec<u8>,
        device: Device,
    ) -> Self {
        let ahb_format = match format {
            ImageFormat::RGB888 => AHARDWAREBUFFER_FORMAT_R8G8B8_UNORM,
            ImageFormat::RGBF32 => AHARDWAREBUFFER_FORMAT_R16G16B16A16_FLOAT,
        };
        let desc = AHardwareBuffer_Desc {
            width,
            height,
            layers: 1,
            format: ahb_format,
            usage: AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE | AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN,
            stride: width,
            rfu0: 0,
            rfu1: 0,
        };

        Self {
            raw_ptr: 0x1 as *mut AHardwareBuffer,
            desc,
            device,
            host_mock_data: Some(data),
        }
    }

    #[inline]
    pub fn raw_ptr(&self) -> *mut AHardwareBuffer {
        self.raw_ptr
    }

    #[inline]
    pub fn desc(&self) -> &AHardwareBuffer_Desc {
        &self.desc
    }

    /// Acquire Vulkan external memory import descriptor for zero-copy GPU compute
    pub fn as_vulkan_external_memory(&self) -> Result<VulkanHardwareBufferImport, AndroidPlatformError> {
        Ok(VulkanHardwareBufferImport::new(
            self.raw_ptr as *mut c_void,
            self.desc.width,
            self.desc.height,
            self.desc.format,
            self.desc.usage,
        ))
    }

    /// Lock the buffer for CPU reading (RAII unlock on drop)
    pub fn lock_cpu_read(&self) -> Result<LockedCpuBuffer<'_>, AndroidPlatformError> {
        if let Some(ref data) = self.host_mock_data {
            return Ok(LockedCpuBuffer {
                buffer: self,
                slice: data.as_slice(),
                _fence: -1,
            });
        }

        #[cfg(target_os = "android")]
        unsafe {
            let mut virtual_addr: *mut c_void = std::ptr::null_mut();
            let status = AHardwareBuffer_lock(
                self.raw_ptr,
                AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN,
                -1,
                std::ptr::null(),
                &mut virtual_addr,
            );
            if status != 0 || virtual_addr.is_null() {
                return Err(AndroidPlatformError::LockFailed(status));
            }

            let len = (self.desc.stride * self.desc.height * 3) as usize;
            let slice = std::slice::from_raw_parts(virtual_addr as *const u8, len);
            Ok(LockedCpuBuffer {
                buffer: self,
                slice,
                _fence: -1,
            })
        }

        #[cfg(not(target_os = "android"))]
        {
            Err(AndroidPlatformError::LockFailed(-1))
        }
    }
}

impl Drop for AndroidHardwareBufferHandle {
    fn drop(&mut self) {
        #[cfg(target_os = "android")]
        if !self.raw_ptr.is_null() {
            unsafe {
                AHardwareBuffer_release(self.raw_ptr);
            }
        }
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
        match self.desc.format {
            AHARDWAREBUFFER_FORMAT_R8G8B8_UNORM | AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM => {
                ImageFormat::RGB888
            }
            AHARDWAREBUFFER_FORMAT_R16G16B16A16_FLOAT => ImageFormat::RGBF32,
            _ => ImageFormat::RGB888,
        }
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn as_bytes(&self) -> Option<&[u8]> {
        self.host_mock_data.as_deref()
    }
}

/// RAII lock for CPU memory reading from an `AHardwareBuffer`
pub struct LockedCpuBuffer<'a> {
    #[allow(dead_code)]
    buffer: &'a AndroidHardwareBufferHandle,
    slice: &'a [u8],
    _fence: i32,
}

impl<'a> LockedCpuBuffer<'a> {
    pub fn as_slice(&self) -> &[u8] {
        self.slice
    }
}

impl<'a> Drop for LockedCpuBuffer<'a> {
    fn drop(&mut self) {
        #[cfg(target_os = "android")]
        if self.buffer.host_mock_data.is_none() && !self.buffer.raw_ptr.is_null() {
            unsafe {
                let mut fence: i32 = -1;
                AHardwareBuffer_unlock(self.buffer.raw_ptr, &mut fence);
            }
        }
    }
}
