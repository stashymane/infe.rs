//! Registration of Infers' [`VulkanContext`] with ExecuTorch's Vulkan runtime.
//!
//! Requires the `vulkan` crate feature and a linked `libvulkan_backend.a`.

use infers_gpu::ash::vk::Handle;
use infers_gpu::VulkanContext;
use std::sync::Arc;

unsafe extern "C" {
    fn infers_et_set_external_vulkan_adapter(
        instance: *mut std::ffi::c_void,
        physical_device: *mut std::ffi::c_void,
        device: *mut std::ffi::c_void,
    ) -> *mut std::ffi::c_void;
}

/// Register `context`'s Vulkan handles with ExecuTorch so the Vulkan delegate
/// reuses the same `VkDevice` (zero-copy-capable sharing).
///
/// Safe to call multiple times with the **same** device; ExecuTorch keeps a
/// process-global sticky adapter after the first successful registration.
pub fn register_external_adapter(context: &Arc<VulkanContext>) -> Result<(), crate::ExecuTorchError> {
    let instance = context.instance_handle().as_raw() as usize as *mut std::ffi::c_void;
    let physical = context.physical_device().as_raw() as usize as *mut std::ffi::c_void;
    let device = context.device_handle().as_raw() as usize as *mut std::ffi::c_void;
    let adapter = unsafe { infers_et_set_external_vulkan_adapter(instance, physical, device) };
    if adapter.is_null() {
        return Err(crate::ExecuTorchError::Execution(
            "Failed to register external Vulkan adapter with ExecuTorch".into(),
        ));
    }
    Ok(())
}
