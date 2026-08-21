use std::ffi::c_void;

/// Descriptor for importing an Android `AHardwareBuffer` into Vulkan via
/// `VK_ANDROID_external_memory_android_hardware_buffer`.
#[derive(Clone, Debug)]
pub struct VulkanHardwareBufferImport {
    pub ahardware_buffer_ptr: *mut c_void,
    pub width: u32,
    pub height: u32,
    pub format: u32,
    pub usage: u64,
}

// Safety: Pointer points to an AHardwareBuffer which is safe to pass across threads
unsafe impl Send for VulkanHardwareBufferImport {}
unsafe impl Sync for VulkanHardwareBufferImport {}

impl VulkanHardwareBufferImport {
    pub fn new(
        ahardware_buffer_ptr: *mut c_void,
        width: u32,
        height: u32,
        format: u32,
        usage: u64,
    ) -> Self {
        Self {
            ahardware_buffer_ptr,
            width,
            height,
            format,
            usage,
        }
    }

    /// Check if buffer usage allows GPU sampled image access
    pub fn supports_gpu_sampling(&self) -> bool {
        (self.usage & crate::ffi::AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE) != 0
    }

    /// Check if buffer usage allows GPU color output access
    pub fn supports_gpu_color_output(&self) -> bool {
        (self.usage & crate::ffi::AHARDWAREBUFFER_USAGE_GPU_COLOR_OUTPUT) != 0
    }
}
