use crate::context::VulkanContext;
use ash::vk;
use infers_core::ImageFormat;
use std::sync::Arc;

/// Vulkan objects and metadata that make up a [`VulkanSampledImage`].
pub struct VulkanSampledImageParts {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub sampler: vk::Sampler,
    pub conversion: Option<vk::SamplerYcbcrConversion>,
    pub is_ycbcr: bool,
    pub acquire_from_external: bool,
    pub width: u32,
    pub height: u32,
    pub format: ImageFormat,
}

/// GPU-resident sampled image (`VkImage` + view + sampler) usable as convert_image input.
pub struct VulkanSampledImage {
    context: Arc<VulkanContext>,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    pub(crate) sampler: vk::Sampler,
    conversion: Option<vk::SamplerYcbcrConversion>,
    is_ycbcr: bool,
    acquire_from_external: bool,
    width: u32,
    height: u32,
    format: ImageFormat,
}

impl VulkanSampledImage {
    pub fn new(context: Arc<VulkanContext>, parts: VulkanSampledImageParts) -> Self {
        Self {
            context,
            image: parts.image,
            memory: parts.memory,
            view: parts.view,
            sampler: parts.sampler,
            conversion: parts.conversion,
            is_ycbcr: parts.is_ycbcr,
            acquire_from_external: parts.acquire_from_external,
            width: parts.width,
            height: parts.height,
            format: parts.format,
        }
    }

    pub fn vulkan_context(&self) -> &Arc<VulkanContext> {
        &self.context
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

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn format(&self) -> ImageFormat {
        self.format
    }
}

impl Drop for VulkanSampledImage {
    fn drop(&mut self) {
        let device = self.context.device();
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
