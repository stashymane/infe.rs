use crate::error::GpuContextError;
use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use infers_core::{Device, DeviceKind};
use std::ffi::CStr;
use std::sync::Mutex;

/// Extra instance/device setup supplied by the app or a platform crate.
#[derive(Clone, Default)]
pub struct VulkanContextOptions {
    pub extra_instance_extensions: Vec<&'static CStr>,
    pub extra_device_extensions: Vec<&'static CStr>,
    pub sampler_ycbcr_conversion: bool,
}

/// Raw Vulkan objects for wrapping an existing device.
pub struct VulkanHandles {
    pub instance: vk::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: vk::Device,
    pub queue: vk::Queue,
    pub queue_family_index: u32,
    pub owns_device: bool,
}

/// Shared Vulkan instance, device, compute queue, and memory allocator.
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
    allocator: Mutex<Option<Allocator>>,
    owns_device: bool,
    logical: Device,
}

impl VulkanContext {
    /// Create a new instance and compute-capable logical device for `device`.
    pub fn new(device: &Device) -> Result<Self, GpuContextError> {
        Self::new_with_options(device, VulkanContextOptions::default())
    }

    pub fn new_with_options(
        device: &Device,
        options: VulkanContextOptions,
    ) -> Result<Self, GpuContextError> {
        if device.kind != DeviceKind::Gpu {
            return Err(GpuContextError::NotGpu(device.clone()));
        }

        let entry = unsafe { ash::Entry::load() }.map_err(|err| GpuContextError::Loader(err.to_string()))?;

        let app_name = std::ffi::CString::new("infers").unwrap();
        let engine_name = std::ffi::CString::new("infers-gpu").unwrap();
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_1);

        let instance_exts: Vec<*const i8> = options
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
            .get(device.id)
            .ok_or(GpuContextError::NoDevice(device.id))?;
        let queue_family_index = first_compute_queue_family(&instance, physical_device)
            .ok_or(GpuContextError::NoDevice(device.id))?;

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

        Ok(Self {
            entry,
            instance,
            physical_device,
            device: created.device,
            queue: created.queue,
            queue_family_index,
            queue_lock: Mutex::new(()),
            allocator: Mutex::new(Some(allocator)),
            owns_device: true,
            logical: device.clone(),
        })
    }

    /// Wrap existing Vulkan handles. Does not create an instance or device.
    pub fn from_raw(logical: Device, handles: VulkanHandles) -> Result<Self, GpuContextError> {
        if logical.kind != DeviceKind::Gpu {
            return Err(GpuContextError::NotGpu(logical));
        }

        let entry = unsafe { ash::Entry::load() }.map_err(|err| GpuContextError::Loader(err.to_string()))?;
        let instance = unsafe { ash::Instance::load(entry.static_fn(), handles.instance) };
        let device = unsafe { ash::Device::load(instance.fp_v1_0(), handles.device) };

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device: handles.physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })?;

        Ok(Self {
            entry,
            instance,
            physical_device: handles.physical_device,
            device,
            queue: handles.queue,
            queue_family_index: handles.queue_family_index,
            queue_lock: Mutex::new(()),
            allocator: Mutex::new(Some(allocator)),
            owns_device: handles.owns_device,
            logical,
        })
    }

    pub fn logical_device(&self) -> &Device {
        &self.logical
    }

    pub fn instance(&self) -> &ash::Instance {
        &self.instance
    }

    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    pub fn device(&self) -> &ash::Device {
        &self.device
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
    ) -> Result<(), GpuContextError> {
        let buffers = [command_buffer];
        let submit = vk::SubmitInfo::default().command_buffers(&buffers);
        let _guard = self.queue_lock.lock().unwrap();
        unsafe {
            self.device.queue_submit(self.queue, &[submit], fence)?;
        }
        Ok(())
    }

    pub fn wait_fence(&self, fence: vk::Fence) -> Result<(), GpuContextError> {
        unsafe {
            self.device.wait_for_fences(&[fence], true, u64::MAX)?;
        }
        Ok(())
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        let _ = unsafe { self.device.device_wait_idle() };
        drop(self.allocator.lock().unwrap().take());
        if self.owns_device {
            unsafe {
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
) -> Result<CreatedDevice, GpuContextError> {
    let available = unsafe { instance.enumerate_device_extension_properties(physical_device) }?;
    let props = unsafe { instance.get_physical_device_properties(physical_device) };
    let api_minor = vk::api_version_minor(props.api_version);

    let mut enabled_exts: Vec<*const i8> = Vec::new();

    let mut features_8bit = vk::PhysicalDevice8BitStorageFeatures::default()
        .storage_buffer8_bit_access(true)
        .uniform_and_storage_buffer8_bit_access(true);
    let mut features_int8 = vk::PhysicalDeviceShaderFloat16Int8Features::default().shader_int8(true);
    let mut features_v12 = vk::PhysicalDeviceVulkan12Features::default()
        .shader_int8(true)
        .storage_buffer8_bit_access(true)
        .uniform_and_storage_buffer8_bit_access(true);
    let mut features_v11 = vk::PhysicalDeviceVulkan11Features::default()
        .sampler_ycbcr_conversion(options.sampler_ycbcr_conversion);

    if api_minor < 2 {
        if !ext_available(&available, vk::KHR_8BIT_STORAGE_NAME) {
            return Err(GpuContextError::MissingFeature("VK_KHR_8bit_storage".into()));
        }
        if !ext_available(&available, vk::KHR_SHADER_FLOAT16_INT8_NAME) {
            return Err(GpuContextError::MissingFeature("VK_KHR_shader_float16_int8".into()));
        }
        enabled_exts.push(vk::KHR_8BIT_STORAGE_NAME.as_ptr());
        enabled_exts.push(vk::KHR_SHADER_FLOAT16_INT8_NAME.as_ptr());
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
            .push_next(&mut features_int8)
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
