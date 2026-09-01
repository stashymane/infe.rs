use crate::error::GpuError;
use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use infers_core::{DeviceInfo, DeviceKind};
use parking_lot::Mutex;
use std::ffi::{CStr, c_char};

/// Extra instance/device setup supplied by the app or a platform crate.
#[derive(Clone, Default)]
pub struct VulkanContextOptions {
    pub extra_instance_extensions: Vec<&'static CStr>,
    pub extra_device_extensions: Vec<&'static CStr>,
    pub sampler_ycbcr_conversion: bool,
}

impl VulkanContextOptions {
    /// Options for a device shared between Infers GPU preprocessing and ExecuTorch Vulkan.
    ///
    /// Device feature bits (Int8, Float16, 16-bit storage) are enabled in
    /// [`VulkanContext::new_with_options`] when the GPU supports them. Platform crates
    /// (e.g. Android) should merge additional extensions on top of this.
    pub fn for_shared_inference() -> Self {
        Self::default()
    }
}

/// Shared Vulkan instance, device, compute queue, and memory allocator.
///
/// Create the context before GPU processors or Vulkan inference sessions, keep an
/// [`Arc`](std::sync::Arc) alive while they run, and drop sessions/processors before the
/// last context handle so the device is destroyed last.
pub struct VulkanContext {
    /// Kept alive so instance/device function pointers remain valid.
    #[allow(dead_code)]
    entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    queue_family_index: u32,
    queue_lock: Mutex<()>,
    oneshot_lock: Mutex<()>,
    oneshot_cmd_pool: vk::CommandPool,
    oneshot_fence: vk::Fence,
    allocator: Mutex<Option<Allocator>>,
    owns_device: bool,
    info: DeviceInfo,
}

impl VulkanContext {
    /// Create a Vulkan context for the GPU at `gpu_id`.
    pub fn new_for_gpu(gpu_id: usize) -> Result<Self, GpuError> {
        let info = DeviceInfo {
            kind: DeviceKind::Gpu,
            id: gpu_id,
            name: format!("GPU:{}", gpu_id),
        };
        Self::new_with_options(&info, VulkanContextOptions::for_shared_inference())
    }

    pub fn new_with_options(
        info: &DeviceInfo,
        options: VulkanContextOptions,
    ) -> Result<Self, GpuError> {
        if info.kind != DeviceKind::Gpu {
            return Err(GpuError::NotGpu(info.clone()));
        }

        let gpu_id = info.id;

        let entry = unsafe { ash::Entry::load() }.map_err(|err| GpuError::Loader(err.to_string()))?;

        let app_name = std::ffi::CString::new("infers").unwrap();
        let engine_name = std::ffi::CString::new("infers-gpu").unwrap();
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_1);

        let instance_exts: Vec<*const c_char> = options
            .extra_instance_extensions
            .iter()
            .map(|name| name.as_ptr())
            .collect();
        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&instance_exts);

        let instance = unsafe { entry.create_instance(&create_info, None) }?;

        let physical_devices = unsafe { instance.enumerate_physical_devices() }?;
        let mut compute_devices = Vec::new();
        for pd in physical_devices {
            if first_compute_queue_family(&instance, pd).is_some() {
                compute_devices.push(pd);
            }
        }
        let physical_device = *compute_devices
            .get(gpu_id)
            .ok_or(GpuError::NoDevice(gpu_id))?;
        let queue_family_index = first_compute_queue_family(&instance, physical_device)
            .ok_or(GpuError::NoDevice(gpu_id))?;

        let created = create_logical_device(
            &instance,
            physical_device,
            queue_family_index,
            &options,
        )?;

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: created.device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })?;

        let oneshot_cmd_pool = unsafe {
            created.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family_index)
                    .flags(
                        vk::CommandPoolCreateFlags::TRANSIENT
                            | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                    ),
                None,
            )?
        };
        let oneshot_fence = unsafe {
            created.device.create_fence(&vk::FenceCreateInfo::default(), None)?
        };

        Ok(Self {
            entry,
            instance,
            physical_device,
            device: created.device,
            queue: created.queue,
            queue_family_index,
            queue_lock: Mutex::new(()),
            oneshot_lock: Mutex::new(()),
            oneshot_cmd_pool,
            oneshot_fence,
            allocator: Mutex::new(Some(allocator)),
            owns_device: true,
            info: info.clone(),
        })
    }

    pub fn device_info(&self) -> &DeviceInfo {
        &self.info
    }

    pub fn instance(&self) -> &ash::Instance {
        &self.instance
    }

    /// Raw `VkInstance` handle for ExecuTorch external-adapter registration.
    pub fn instance_handle(&self) -> vk::Instance {
        self.instance.handle()
    }

    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    /// Raw `VkDevice` handle for ExecuTorch external-adapter registration.
    pub fn device_handle(&self) -> vk::Device {
        self.device.handle()
    }

    pub fn queue(&self) -> vk::Queue {
        self.queue
    }

    pub fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }

    pub fn allocator(&self) -> &Mutex<Option<Allocator>> {
        &self.allocator
    }

    pub fn submit(
        &self,
        command_buffer: vk::CommandBuffer,
        fence: vk::Fence,
    ) -> Result<(), GpuError> {
        let buffers = [command_buffer];
        let submit = vk::SubmitInfo::default().command_buffers(&buffers);
        // Vulkan queues are not thread-safe; serialise submissions on this one.
        let _guard = self.queue_lock.lock();
        // SAFETY: `command_buffer` and `fence` belong to this device, the queue
        // is exclusively held for this call, and `submit` borrows `buffers`,
        // which outlives it.
        unsafe {
            self.device.queue_submit(self.queue, &[submit], fence)?;
        }
        Ok(())
    }

    pub fn wait_fence(&self, fence: vk::Fence) -> Result<(), GpuError> {
        // SAFETY: `fence` belongs to this device and waiting on it is valid from
        // any thread.
        unsafe {
            self.device.wait_for_fences(&[fence], true, u64::MAX)?;
        }
        Ok(())
    }

    /// Record a one-shot command buffer with `record`, submit it, and block until
    /// the GPU has finished.
    pub fn record_and_wait<F>(&self, record: F) -> Result<(), GpuError>
    where
        F: FnOnce(&ash::Device, vk::CommandBuffer) -> Result<(), GpuError>,
    {
        let _guard = self.oneshot_lock.lock();
        let device = self.device();

        // SAFETY: `oneshot_cmd_pool` belongs to this device and is exclusively
        // held under `oneshot_lock`.
        unsafe {
            device.reset_command_pool(
                self.oneshot_cmd_pool,
                vk::CommandPoolResetFlags::empty(),
            )?;
            device.reset_fences(&[self.oneshot_fence])?;
        }

        let cmd = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.oneshot_cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?
        }[0];

        // SAFETY: `cmd` was freshly allocated and is not recording or pending.
        unsafe {
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }

        record(device, cmd)?;

        // SAFETY: `cmd` is in the recording state, opened just above.
        unsafe { device.end_command_buffer(cmd)? };

        self.submit(cmd, self.oneshot_fence)?;
        self.wait_fence(self.oneshot_fence)?;
        Ok(())
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        // SAFETY: waiting for idle before teardown ensures no queued work still
        // references the allocator's memory or the device itself.
        let _ = unsafe { self.device.device_wait_idle() };
        // The allocator must release its memory before the device is destroyed.
        drop(self.allocator.lock().take());
        if self.owns_device {
            // SAFETY: this context created both objects and, per its documented
            // contract, outlives every buffer, image and session using them. The
            // device is destroyed before the instance that created it.
            unsafe {
                self.device.destroy_fence(self.oneshot_fence, None);
                self.device.destroy_command_pool(self.oneshot_cmd_pool, None);
                self.device.destroy_device(None);
                self.instance.destroy_instance(None);
            }
        }
    }
}

fn first_compute_queue_family(instance: &ash::Instance, pd: vk::PhysicalDevice) -> Option<u32> {
    let props = unsafe { instance.get_physical_device_queue_family_properties(pd) };
    props.iter().enumerate().find_map(|(i, q)| {
        if q.queue_flags.contains(vk::QueueFlags::COMPUTE) {
            Some(i as u32)
        } else {
            None
        }
    })
}

fn ext_available(available: &[vk::ExtensionProperties], name: &CStr) -> bool {
    available.iter().any(|e| e.extension_name_as_c_str() == Ok(name))
}

struct CreatedDevice {
    device: ash::Device,
    queue: vk::Queue,
}

fn create_logical_device(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
    options: &VulkanContextOptions,
) -> Result<CreatedDevice, GpuError> {
    let available = unsafe { instance.enumerate_device_extension_properties(physical_device) }?;
    let props = unsafe { instance.get_physical_device_properties(physical_device) };
    let api_minor = vk::api_version_minor(props.api_version);

    // `c_char` is signed on most targets but unsigned on Android/aarch64, so the
    // element type must follow the platform rather than being spelled `i8`.
    let mut enabled_exts: Vec<*const c_char> = Vec::new();

    let mut features_8bit = vk::PhysicalDevice8BitStorageFeatures::default()
        .storage_buffer8_bit_access(true)
        .uniform_and_storage_buffer8_bit_access(true);
    let mut features_16bit = vk::PhysicalDevice16BitStorageFeatures::default()
        .storage_buffer16_bit_access(true)
        .uniform_and_storage_buffer16_bit_access(true);
    let mut features_float16_int8 = vk::PhysicalDeviceShaderFloat16Int8Features::default()
        .shader_int8(true)
        .shader_float16(true);
    let mut features_v12 = vk::PhysicalDeviceVulkan12Features::default()
        .shader_int8(true)
        .shader_float16(true)
        .storage_buffer8_bit_access(true)
        .uniform_and_storage_buffer8_bit_access(true);
    let mut features_v11 = vk::PhysicalDeviceVulkan11Features::default()
        .sampler_ycbcr_conversion(options.sampler_ycbcr_conversion)
        .storage_buffer16_bit_access(true)
        .uniform_and_storage_buffer16_bit_access(true);

    if api_minor < 2 {
        if !ext_available(&available, vk::KHR_8BIT_STORAGE_NAME) {
            return Err(GpuError::MissingFeature("VK_KHR_8bit_storage".into()));
        }
        if !ext_available(&available, vk::KHR_SHADER_FLOAT16_INT8_NAME) {
            return Err(GpuError::MissingFeature("VK_KHR_shader_float16_int8".into()));
        }
        enabled_exts.push(vk::KHR_8BIT_STORAGE_NAME.as_ptr());
        enabled_exts.push(vk::KHR_SHADER_FLOAT16_INT8_NAME.as_ptr());
        if ext_available(&available, vk::KHR_16BIT_STORAGE_NAME) {
            enabled_exts.push(vk::KHR_16BIT_STORAGE_NAME.as_ptr());
        }
    }

    for name in &options.extra_device_extensions {
        if ext_available(&available, name) {
            enabled_exts.push(name.as_ptr());
        }
    }

    let priorities = [1.0f32];
    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&priorities);

    let mut features2 = vk::PhysicalDeviceFeatures2::default();
    if api_minor >= 2 {
        features2 = features2
            .push_next(&mut features_v12)
            .push_next(&mut features_v11);
    } else {
        features2 = features2
            .push_next(&mut features_8bit)
            .push_next(&mut features_float16_int8)
            .push_next(&mut features_16bit)
            .push_next(&mut features_v11);
    }

    let queue_infos = [queue_info];
    let device_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&enabled_exts)
        .push_next(&mut features2);

    let device = unsafe { instance.create_device(physical_device, &device_info, None) }?;
    let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

    Ok(CreatedDevice { device, queue })
}
