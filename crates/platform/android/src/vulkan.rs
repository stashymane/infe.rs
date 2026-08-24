use crate::error::AndroidPlatformError;
use crate::hardware_buffer::AndroidHardwareBufferHandle;
use infers_core::Device;
use infers_gpu::ash::vk;
use infers_gpu::{VulkanContext, VulkanContextOptions, VulkanSampledImage};
use std::sync::Arc;

/// Instance/device extensions required to import `AHardwareBuffer` into Vulkan.
pub fn vulkan_context_options() -> VulkanContextOptions {
    VulkanContextOptions {
        extra_instance_extensions: vec![
            ash::khr::external_memory_capabilities::NAME,
            ash::khr::get_physical_device_properties2::NAME,
        ],
        extra_device_extensions: vec![
            ash::android::external_memory_android_hardware_buffer::NAME,
            ash::khr::external_memory::NAME,
            ash::ext::queue_family_foreign::NAME,
        ],
        sampler_ycbcr_conversion: true,
    }
}

pub fn create_vulkan_context(device: &Device) -> Result<VulkanContext, AndroidPlatformError> {
    VulkanContext::new_with_options(device, vulkan_context_options()).map_err(|err| {
        AndroidPlatformError::VulkanImportError(err.to_string())
    })
}

impl AndroidHardwareBufferHandle {
    /// Import this hardware buffer as a sampled Vulkan image on `context`.
    pub fn to_vulkan(
        &self,
        context: Arc<VulkanContext>,
    ) -> Result<VulkanSampledImage, AndroidPlatformError> {
        import_hardware_buffer(context, self)
    }
}

fn import_hardware_buffer(
    context: Arc<VulkanContext>,
    src: &AndroidHardwareBufferHandle,
) -> Result<VulkanSampledImage, AndroidPlatformError> {
    let ahb = ash::android::external_memory_android_hardware_buffer::Device::new(
        context.instance(),
        context.device(),
    );

    let mut format_props = vk::AndroidHardwareBufferFormatPropertiesANDROID::default();
    let mut props =
        vk::AndroidHardwareBufferPropertiesANDROID::default().push_next(&mut format_props);
    unsafe {
        ahb.get_android_hardware_buffer_properties(src.raw_ptr().cast(), &mut props)
            .map_err(|err| AndroidPlatformError::VulkanImportError(err.to_string()))?;
    }

    let is_ycbcr = format_props.external_format != 0
        || matches!(
            format_props.format,
            vk::Format::G8_B8R8_2PLANE_420_UNORM | vk::Format::G8_B8_R8_3PLANE_420_UNORM
        );

    let mut ext_mem = vk::ExternalMemoryImageCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::ANDROID_HARDWARE_BUFFER_ANDROID);

    let mut ext_format =
        vk::ExternalFormatANDROID::default().external_format(format_props.external_format);

    let mut image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(if format_props.external_format != 0 {
            vk::Format::UNDEFINED
        } else {
            format_props.format
        })
        .extent(vk::Extent3D {
            width: src.width(),
            height: src.height(),
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .push_next(&mut ext_mem);

    if format_props.external_format != 0 {
        image_info = image_info.push_next(&mut ext_format);
    }

    let device = context.device();
    let image = unsafe {
        device
            .create_image(&image_info, None)
            .map_err(|err| AndroidPlatformError::VulkanImportError(err.to_string()))?
    };

    let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
    let mut import =
        vk::ImportAndroidHardwareBufferInfoANDROID::default().buffer(src.raw_ptr().cast());
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(props.allocation_size)
        .memory_type_index(memory_type_index(
            &context,
            props.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?)
        .push_next(&mut dedicated)
        .push_next(&mut import);

    let memory = unsafe {
        device
            .allocate_memory(&alloc_info, None)
            .map_err(|err| AndroidPlatformError::VulkanImportError(err.to_string()))?
    };
    unsafe {
        device
            .bind_image_memory(image, memory, 0)
            .map_err(|err| AndroidPlatformError::VulkanImportError(err.to_string()))?;
    }

    let mut conversion = None;
    let mut ycbcr_sampler_info = vk::SamplerYcbcrConversionInfo::default();
    let mut ycbcr_view_info = vk::SamplerYcbcrConversionInfo::default();
    if is_ycbcr {
        let mut conv_info = vk::SamplerYcbcrConversionCreateInfo::default()
            .format(if format_props.external_format != 0 {
                vk::Format::UNDEFINED
            } else {
                format_props.format
            })
            .ycbcr_model(format_props.suggested_ycbcr_model)
            .ycbcr_range(format_props.suggested_ycbcr_range)
            .components(format_props.sampler_ycbcr_conversion_components)
            .x_chroma_offset(format_props.suggested_x_chroma_offset)
            .y_chroma_offset(format_props.suggested_y_chroma_offset)
            .chroma_filter(vk::Filter::LINEAR)
            .force_explicit_reconstruction(false);
        let mut conv_ext =
            vk::ExternalFormatANDROID::default().external_format(format_props.external_format);
        if format_props.external_format != 0 {
            conv_info = conv_info.push_next(&mut conv_ext);
        }
        let conv = unsafe {
            device
                .create_sampler_ycbcr_conversion(&conv_info, None)
                .map_err(|err| AndroidPlatformError::VulkanImportError(err.to_string()))?
        };
        conversion = Some(conv);
        ycbcr_sampler_info = vk::SamplerYcbcrConversionInfo::default().conversion(conv);
        ycbcr_view_info = vk::SamplerYcbcrConversionInfo::default().conversion(conv);
    }

    let mut sampler_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
    if is_ycbcr {
        sampler_info = sampler_info.push_next(&mut ycbcr_sampler_info);
    }
    let sampler = unsafe {
        device
            .create_sampler(&sampler_info, None)
            .map_err(|err| AndroidPlatformError::VulkanImportError(err.to_string()))?
    };

    let mut view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(if format_props.external_format != 0 {
            vk::Format::UNDEFINED
        } else {
            format_props.format
        })
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    if is_ycbcr {
        view_info = view_info.push_next(&mut ycbcr_view_info);
    }
    let view = unsafe {
        device
            .create_image_view(&view_info, None)
            .map_err(|err| AndroidPlatformError::VulkanImportError(err.to_string()))?
    };

    Ok(VulkanSampledImage::new(
        context,
        image,
        memory,
        view,
        sampler,
        conversion,
        is_ycbcr,
        true,
        src.width(),
        src.height(),
        src.format(),
        src.device().clone(),
    ))
}

fn memory_type_index(
    context: &VulkanContext,
    type_bits: u32,
    required: vk::MemoryPropertyFlags,
) -> Result<u32, AndroidPlatformError> {
    let mems = unsafe {
        context
            .instance()
            .get_physical_device_memory_properties(context.physical_device())
    };
    for i in 0..mems.memory_type_count {
        if type_bits & (1 << i) != 0
            && mems.memory_types[i as usize]
                .property_flags
                .contains(required)
        {
            return Ok(i);
        }
    }
    for i in 0..mems.memory_type_count {
        if type_bits & (1 << i) != 0 {
            return Ok(i);
        }
    }
    Err(AndroidPlatformError::VulkanImportError(
        "no compatible memory type for AHardwareBuffer".into(),
    ))
}
