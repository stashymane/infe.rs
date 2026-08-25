//! Registration of Infers' [`VulkanContext`] with ExecuTorch's Vulkan runtime.
//!
//! Requires the `vulkan` crate feature and a linked `libvulkan_backend.a`.
//!
//! ExecuTorch keeps a **single** process-global sticky external adapter. Destroy
//! it via [`ExternalAdapterRegistration`]'s `Drop` when the last registration
//! ends, **while** the shared [`VulkanContext`] is still alive, and only after
//! all sessions / `Module`s that reference the adapter have been dropped.

use infers_gpu::ash::vk::Handle;
use infers_gpu::VulkanContext;
use std::sync::{Arc, Mutex, OnceLock};

unsafe extern "C" {
    fn infers_et_set_external_vulkan_adapter(
        instance: *mut std::ffi::c_void,
        physical_device: *mut std::ffi::c_void,
        device: *mut std::ffi::c_void,
    ) -> *mut std::ffi::c_void;

    fn infers_et_clear_external_vulkan_adapter();
}

struct RegistrationState {
    device_raw: Option<u64>,
    count: usize,
}

fn registration_state() -> &'static Mutex<RegistrationState> {
    static STATE: OnceLock<Mutex<RegistrationState>> = OnceLock::new();
    STATE.get_or_init(|| {
        Mutex::new(RegistrationState {
            device_raw: None,
            count: 0,
        })
    })
}

/// RAII guard for one live registration of the process-global external adapter.
///
/// Dropping the last guard calls `clear_external_adapter` while this guard still
/// holds an [`Arc`] to the context, so VMA / pipeline caches tear down against a
/// valid device.
pub struct ExternalAdapterRegistration {
    context: Arc<VulkanContext>,
}

impl ExternalAdapterRegistration {
    pub fn context(&self) -> &Arc<VulkanContext> {
        &self.context
    }
}

impl Drop for ExternalAdapterRegistration {
    fn drop(&mut self) {
        let should_clear = {
            let Ok(mut state) = registration_state().lock() else {
                return;
            };
            if state.count > 0 {
                state.count -= 1;
            }
            let clear = state.count == 0;
            if clear {
                state.device_raw = None;
            }
            clear
        };
        if should_clear {
            // SAFETY: no live ExecuTorch Vulkan sessions should hold the sticky
            // adapter; callers drop Modules before this guard. The VkDevice is
            // still valid via `self.context`.
            unsafe { infers_et_clear_external_vulkan_adapter() };
        }
    }
}

/// Register `context`'s Vulkan handles with ExecuTorch so the Vulkan delegate
/// reuses the same `VkDevice` (zero-copy-capable sharing).
///
/// Returns a guard that must outlive any `Module` / session using the adapter,
/// and must be dropped **before** the last [`Arc`] to `context` is released
/// (the guard itself retains one clone until clear completes).
///
/// ExecuTorch only supports one sticky external adapter per process. Registering
/// a different `VkDevice` while another registration is live returns an error.
pub fn register_external_adapter(
    context: &Arc<VulkanContext>,
) -> Result<ExternalAdapterRegistration, crate::ExecuTorchError> {
    let device_raw = context.device_handle().as_raw();
    {
        let state = registration_state().lock().map_err(|_| {
            crate::ExecuTorchError::Execution("Vulkan registration lock poisoned".into())
        })?;
        if let Some(existing) = state.device_raw {
            if existing != device_raw {
                return Err(crate::ExecuTorchError::Execution(
                    "ExecuTorch external Vulkan adapter is already registered for a different VkDevice"
                        .into(),
                ));
            }
        }
    }

    let instance = context.instance_handle().as_raw() as usize as *mut std::ffi::c_void;
    let physical = context.physical_device().as_raw() as usize as *mut std::ffi::c_void;
    let device = context.device_handle().as_raw() as usize as *mut std::ffi::c_void;
    let adapter = unsafe { infers_et_set_external_vulkan_adapter(instance, physical, device) };
    if adapter.is_null() {
        return Err(crate::ExecuTorchError::Execution(
            "Failed to register external Vulkan adapter with ExecuTorch".into(),
        ));
    }

    {
        let mut state = registration_state().lock().map_err(|_| {
            crate::ExecuTorchError::Execution("Vulkan registration lock poisoned".into())
        })?;
        state.device_raw = Some(device_raw);
        state.count += 1;
    }

    Ok(ExternalAdapterRegistration {
        context: Arc::clone(context),
    })
}
