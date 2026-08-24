use crate::context::VulkanContext;
use ash::vk;
use infers_core::{Device, ImageFormat, ImageInputBuffer};
use std::any::Any;
use std::sync::Arc;

/// GPU-resident sampled image (`VkImage` + view + sampler) usable as convert_image input.
pub struct VulkanSampledImage {
    context: Arc<VulkanContext>,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    sampler: vk::Sampler,
    conversion: Option<vk::SamplerYcbcrConversion>,
    is_ycbcr: bool,
    acquire_from_external: bool,
    width: u32,
    height: u32,
    format: ImageFormat,
    device: Device,
}

impl VulkanSampledImage {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context: Arc<VulkanContext>,
        image: vk::Image,
        memory: vk::DeviceMemory,
        view: vk::ImageView,
        sampler: vk::Sampler,
        conversion: Option<vk::SamplerYcbcrConversion>,
        is_ycbcr: bool,
        acquire_from_external: bool,
        width: u32,
        height: u32,
        format: ImageFormat,
        device: Device,
    ) -> Self {
        Self {
            context,
            image,
            memory,
            view,
            sampler,
            conversion,
            is_ycbcr,
            acquire_from_external,
            width,
            height,
            format,
            device,
        }
    }

    pub fn image(&self) -> vk::Image {
        self.image
    }

    pub fn view(&self) -> vk::ImageView {
        self.view
    }

    pub fn sampler(&self) -> vk::Sampler {
        self.sampler
    }

    pub fn is_ycbcr(&self) -> bool {
        self.is_ycbcr
    }

    pub fn acquire_from_external(&self) -> bool {
        self.acquire_from_external
    }
}

impl Drop for VulkanSampledImage {
    fn drop(&mut self) {
        let device = self.context.device();
        let _ = unsafe { device.device_wait_idle() };
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_sampler(self.sampler, None);
            if let Some(conv) = self.conversion {
                device.destroy_sampler_ycbcr_conversion(conv, None);
            }
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}

impl ImageInputBuffer for VulkanSampledImage {
    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
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

    fn as_any(&self) -> &dyn Any {
        self
    }
}
