use crate::context::VulkanContext;
use ash::vk;
use infers_core::{Device, ImageFormat, ImageInputBuffer};
use std::any::Any;
use std::sync::Arc;

/// Vulkan objects and metadata that make up a [`VulkanSampledImage`].
///
/// Grouped into a struct so callers name each handle at the construction site;
/// the handles are otherwise interchangeable at the type level and easy to
/// transpose.
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
    pub device: Device,
}

/// GPU-resident sampled image (`VkImage` + view + sampler) usable as convert_image input.
///
/// Takes ownership of every handle in [`VulkanSampledImageParts`] and destroys
/// them on drop.
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
    /// Take ownership of `parts`, which must have been created from `context`.
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
            device: parts.device,
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
        // Callers must not drop this image until every submitted command that
        // reads it has completed (e.g. `GpuImageProcessor::process` waits on its
        // fence before returning). `VulkanContext` still waits idle on teardown.
        let device = self.context.device();
        // SAFETY: this image owns every handle below, each created from
        // `self.context`'s device and destroyed exactly once here.
        // Destruction order respects Vulkan's dependencies: the view and sampler
        // reference the image and the ycbcr conversion, and the image must be
        // destroyed before the memory backing it is freed.
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
