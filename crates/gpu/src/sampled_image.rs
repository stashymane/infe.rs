use crate::context::VulkanContext;
use crate::ycbcr::SharedYcbcrSampler;
use ash::vk;
use infers_core::ImageFormat;
use std::sync::Arc;
use std::sync::atomic::{AtomicI32, Ordering};

/// Vulkan objects and metadata that make up a [`VulkanSampledImage`].
pub struct VulkanSampledImageParts {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    /// Non-YCbCr sampler owned by this image, or null when [`ycbcr`] is set.
    pub sampler: vk::Sampler,
    /// Shared YCbCr conversion+sampler (camera / external formats).
    pub ycbcr: Option<Arc<SharedYcbcrSampler>>,
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
    /// Owned linear sampler when not YCbCr.
    owned_sampler: vk::Sampler,
    ycbcr: Option<Arc<SharedYcbcrSampler>>,
    is_ycbcr: bool,
    acquire_from_external: bool,
    width: u32,
    height: u32,
    format: ImageFormat,
    layout: AtomicI32,
}

impl VulkanSampledImage {
    pub fn new(context: Arc<VulkanContext>, parts: VulkanSampledImageParts) -> Self {
        Self {
            context,
            image: parts.image,
            memory: parts.memory,
            view: parts.view,
            owned_sampler: parts.sampler,
            ycbcr: parts.ycbcr,
            is_ycbcr: parts.is_ycbcr,
            acquire_from_external: parts.acquire_from_external,
            width: parts.width,
            height: parts.height,
            format: parts.format,
            layout: AtomicI32::new(vk::ImageLayout::UNDEFINED.as_raw()),
        }
    }

    pub fn current_layout(&self) -> vk::ImageLayout {
        vk::ImageLayout::from_raw(self.layout.load(Ordering::Relaxed))
    }

    pub fn set_layout(&self, layout: vk::ImageLayout) {
        self.layout.store(layout.as_raw(), Ordering::Relaxed);
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
        if let Some(ycbcr) = &self.ycbcr {
            ycbcr.sampler()
        } else {
            self.owned_sampler
        }
    }

    pub fn shared_ycbcr(&self) -> Option<&Arc<SharedYcbcrSampler>> {
        self.ycbcr.as_ref()
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
            if self.owned_sampler != vk::Sampler::null() {
                device.destroy_sampler(self.owned_sampler, None);
            }
            // YCbCr conversion+sampler are owned by `SharedYcbcrSampler`.
            self.ycbcr = None;
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}
