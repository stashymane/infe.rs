use ash::vk;
use ash::vk::Handle;
use std::sync::Arc;

/// Identity for a reusable [`SharedYcbcrSampler`].
///
/// Camera frames that share the same external/YUV format must reuse one
/// conversion + sampler so immutable-sampler descriptor layouts stay stable.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct YcbcrConversionKey {
    pub format: i32,
    pub external_format: u64,
    pub model: i32,
    pub range: i32,
    pub swizzle_r: i32,
    pub swizzle_g: i32,
    pub swizzle_b: i32,
    pub swizzle_a: i32,
    pub x_chroma_offset: i32,
    pub y_chroma_offset: i32,
}

impl YcbcrConversionKey {
    pub fn from_parts(
        format: vk::Format,
        external_format: u64,
        model: vk::SamplerYcbcrModelConversion,
        range: vk::SamplerYcbcrRange,
        components: vk::ComponentMapping,
        x_chroma_offset: vk::ChromaLocation,
        y_chroma_offset: vk::ChromaLocation,
    ) -> Self {
        Self {
            format: format.as_raw(),
            external_format,
            model: model.as_raw(),
            range: range.as_raw(),
            swizzle_r: components.r.as_raw(),
            swizzle_g: components.g.as_raw(),
            swizzle_b: components.b.as_raw(),
            swizzle_a: components.a.as_raw(),
            x_chroma_offset: x_chroma_offset.as_raw(),
            y_chroma_offset: y_chroma_offset.as_raw(),
        }
    }
}

/// Shared `VkSamplerYcbcrConversion` + `VkSampler` owned by the device cache.
///
/// Descriptor set layouts with immutable YCbCr samplers require the sampler to
/// outlive the layout; caching keeps one handle per format instead of one per
/// frame.
pub struct SharedYcbcrSampler {
    device: ash::Device,
    conversion: vk::SamplerYcbcrConversion,
    sampler: vk::Sampler,
}

impl SharedYcbcrSampler {
    pub fn new(
        device: ash::Device,
        conversion: vk::SamplerYcbcrConversion,
        sampler: vk::Sampler,
    ) -> Arc<Self> {
        Arc::new(Self {
            device,
            conversion,
            sampler,
        })
    }

    pub fn conversion(&self) -> vk::SamplerYcbcrConversion {
        self.conversion
    }

    pub fn sampler(&self) -> vk::Sampler {
        self.sampler
    }

    pub fn sampler_key(&self) -> u64 {
        self.sampler.as_raw()
    }
}

impl Drop for SharedYcbcrSampler {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_sampler(self.sampler, None);
            self.device
                .destroy_sampler_ycbcr_conversion(self.conversion, None);
        }
    }
}
