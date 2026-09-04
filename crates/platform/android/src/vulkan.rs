use crate::error::AndroidPlatformError;
use crate::hardware_buffer::AndroidHardwareBufferHandle;
use infers_core::DeviceInfo;
use infers_gpu::ash::vk;
use infers_gpu::{
    VulkanContext, VulkanContextOptions, VulkanSampledImage, VulkanSampledImageParts,
};
use std::sync::Arc;

/// Instance/device extensions required to import `AHardwareBuffer` into Vulkan,
/// merged with shared-inference defaults for ExecuTorch Vulkan.
pub fn vulkan_context_options() -> VulkanContextOptions {
    let mut options = VulkanContextOptions::for_shared_inference();
    options.extra_instance_extensions = vec![
        ash::khr::external_memory_capabilities::NAME,
        ash::khr::get_physical_device_properties2::NAME,
    ];
    options.extra_device_extensions = vec![
        ash::android::external_memory_android_hardware_buffer::NAME,
        ash::khr::external_memory::NAME,
        ash::ext::queue_family_foreign::NAME,
    ];
    options.sampler_ycbcr_conversion = true;
    options
}

pub fn create_vulkan_context(device: &DeviceInfo) -> Result<VulkanContext, AndroidPlatformError> {
    VulkanContext::new_with_options(device, vulkan_context_options()).map_err(|err| {
        AndroidPlatformError::VulkanImportError(err.to_string())
    })
}

impl AndroidHardwareBufferHandle {
    /// Defer zero-copy Vulkan import until materialize/process time.
    pub fn on(&self, vulkan: &infers_gpu::Vulkan) -> infers_core::Deferred<infers_gpu::Vulkan> {
        use infers_core::CoreError;
        use infers_gpu::VulkanImage;
        use std::sync::Arc;

        let owned = self.clone();
        infers_core::Deferred::from_custom(vulkan.clone(), move |device| {
            let sampled = owned
                .to_vulkan(Arc::clone(device.context()))
                .map_err(|err| CoreError::Platform(err.to_string()))?;
            Ok(VulkanImage::Sampled(std::sync::Arc::new(sampled)))
        })
    }

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

    let queried = query_buffer_properties(&ahb, src)?;

    let is_ycbcr = queried.external_format != 0
        || matches!(
            queried.format,
            vk::Format::G8_B8R8_2PLANE_420_UNORM | vk::Format::G8_B8_R8_3PLANE_420_UNORM
        );

    // An external format is opaque to Vulkan, so the image, view and conversion
    // must all be created with `UNDEFINED` and rely on the `ExternalFormatANDROID`
    // chain instead.
    let view_format = if queried.external_format != 0 {
        vk::Format::UNDEFINED
    } else {
        queried.format
    };

    let mut ext_mem = vk::ExternalMemoryImageCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::ANDROID_HARDWARE_BUFFER_ANDROID);

    let mut ext_format =
        vk::ExternalFormatANDROID::default().external_format(queried.external_format);

    let mut image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(view_format)
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

    if queried.external_format != 0 {
        image_info = image_info.push_next(&mut ext_format);
    }

    let device = context.device();
    // SAFETY: `image_info` is fully initialised above and its `push_next` chain
    // borrows `ext_mem`/`ext_format`, both alive until this call returns.
    let image = unsafe {
        device
            .create_image(&image_info, None)
            .map_err(|err| AndroidPlatformError::VulkanImportError(err.to_string()))?
    };

    // Every handle created past this point is registered with the guard, so an
    // error on any later step destroys the partially-built image instead of
    // leaking it. Disarmed once ownership transfers to the `VulkanSampledImage`.
    let mut guard = ImportGuard {
        device,
        image,
        memory: vk::DeviceMemory::null(),
        conversion: None,
        sampler: vk::Sampler::null(),
        view: vk::ImageView::null(),
    };

    let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
    let mut import =
        vk::ImportAndroidHardwareBufferInfoANDROID::default().buffer(src.raw_ptr().cast());
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(queried.allocation_size)
        .memory_type_index(memory_type_index(
            &context,
            queried.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?)
        .push_next(&mut dedicated)
        .push_next(&mut import);

    // SAFETY: `alloc_info` is fully initialised and its `push_next` chain borrows
    // `dedicated`/`import`, both alive until this call returns.
    let memory = unsafe {
        device
            .allocate_memory(&alloc_info, None)
            .map_err(|err| AndroidPlatformError::VulkanImportError(err.to_string()))?
    };
    guard.memory = memory;
    // SAFETY: `image` has no memory bound yet and `memory` was just allocated
    // with a dedicated-allocation info naming that same image.
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
            .format(view_format)
            .ycbcr_model(queried.suggested_ycbcr_model)
            .ycbcr_range(queried.suggested_ycbcr_range)
            .components(queried.sampler_ycbcr_conversion_components)
            .x_chroma_offset(queried.suggested_x_chroma_offset)
            .y_chroma_offset(queried.suggested_y_chroma_offset)
            .chroma_filter(vk::Filter::LINEAR)
            .force_explicit_reconstruction(false);
        let mut conv_ext =
            vk::ExternalFormatANDROID::default().external_format(queried.external_format);
        if queried.external_format != 0 {
            conv_info = conv_info.push_next(&mut conv_ext);
        }
        // SAFETY: `conv_info` is fully initialised and any `push_next` borrow
        // (`conv_ext`) is alive until this call returns.
        let conv = unsafe {
            device
                .create_sampler_ycbcr_conversion(&conv_info, None)
                .map_err(|err| AndroidPlatformError::VulkanImportError(err.to_string()))?
        };
        guard.conversion = Some(conv);
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
    // SAFETY: `sampler_info` is fully initialised and any `push_next` borrow
    // (`ycbcr_sampler_info`) is alive until this call returns.
    let sampler = unsafe {
        device
            .create_sampler(&sampler_info, None)
            .map_err(|err| AndroidPlatformError::VulkanImportError(err.to_string()))?
    };
    guard.sampler = sampler;

    let mut view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(view_format)
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
    // SAFETY: `view_info` is fully initialised, names the image created above,
    // and any `push_next` borrow (`ycbcr_view_info`) is alive until this returns.
    let view = unsafe {
        device
            .create_image_view(&view_info, None)
            .map_err(|err| AndroidPlatformError::VulkanImportError(err.to_string()))?
    };
    guard.view = view;

    // Nothing can fail past this point, so hand the handles to the image, which
    // becomes responsible for destroying them.
    guard.disarm();

    Ok(VulkanSampledImage::new(
        context,
        VulkanSampledImageParts {
            image,
            memory,
            view,
            sampler,
            conversion,
            is_ycbcr,
            acquire_from_external: true,
            width: src.width(),
            height: src.height(),
            format: src.format(),
        },
    ))
}

/// Everything the import needs from `vkGetAndroidHardwareBufferPropertiesANDROID`.
///
/// The `ash` builders chain `AndroidHardwareBufferFormatPropertiesANDROID` into
/// `AndroidHardwareBufferPropertiesANDROID` by mutable borrow, which would keep
/// the format struct borrowed for as long as the outer struct lives. Copying the
/// plain-old-data fields out releases both borrows immediately.
struct QueriedBufferProperties {
    allocation_size: vk::DeviceSize,
    memory_type_bits: u32,
    format: vk::Format,
    external_format: u64,
    suggested_ycbcr_model: vk::SamplerYcbcrModelConversion,
    suggested_ycbcr_range: vk::SamplerYcbcrRange,
    sampler_ycbcr_conversion_components: vk::ComponentMapping,
    suggested_x_chroma_offset: vk::ChromaLocation,
    suggested_y_chroma_offset: vk::ChromaLocation,
}

fn query_buffer_properties(
    ahb: &ash::android::external_memory_android_hardware_buffer::Device,
    src: &AndroidHardwareBufferHandle,
) -> Result<QueriedBufferProperties, AndroidPlatformError> {
    let mut format_props = vk::AndroidHardwareBufferFormatPropertiesANDROID::default();
    let mut props =
        vk::AndroidHardwareBufferPropertiesANDROID::default().push_next(&mut format_props);
    // SAFETY: `src` owns a live `AHardwareBuffer` for the duration of this call
    // and `props` is a valid out-parameter whose `push_next` chain borrows
    // `format_props`, which outlives the call.
    unsafe {
        ahb.get_android_hardware_buffer_properties(src.raw_ptr().cast(), &mut props)
            .map_err(|err| AndroidPlatformError::VulkanImportError(err.to_string()))?;
    }

    // Reading these two fields is the last use of `props`, which ends its
    // mutable borrow of `format_props` and lets the rest be copied out below.
    let allocation_size = props.allocation_size;
    let memory_type_bits = props.memory_type_bits;

    Ok(QueriedBufferProperties {
        allocation_size,
        memory_type_bits,
        format: format_props.format,
        external_format: format_props.external_format,
        suggested_ycbcr_model: format_props.suggested_ycbcr_model,
        suggested_ycbcr_range: format_props.suggested_ycbcr_range,
        sampler_ycbcr_conversion_components: format_props.sampler_ycbcr_conversion_components,
        suggested_x_chroma_offset: format_props.suggested_x_chroma_offset,
        suggested_y_chroma_offset: format_props.suggested_y_chroma_offset,
    })
}

/// Destroys the Vulkan handles built so far unless [`ImportGuard::disarm`] is
/// called, so a failure part-way through the import does not leak them.
struct ImportGuard<'a> {
    device: &'a ash::Device,
    image: vk::Image,
    memory: vk::DeviceMemory,
    conversion: Option<vk::SamplerYcbcrConversion>,
    sampler: vk::Sampler,
    view: vk::ImageView,
}

impl ImportGuard<'_> {
    fn disarm(mut self) {
        self.image = vk::Image::null();
        self.memory = vk::DeviceMemory::null();
        self.conversion = None;
        self.sampler = vk::Sampler::null();
        self.view = vk::ImageView::null();
    }
}

impl Drop for ImportGuard<'_> {
    fn drop(&mut self) {
        // SAFETY: each non-null handle was created from `self.device` earlier in
        // the import, is not owned by anything else (the guard is disarmed before
        // ownership transfers), and has never been submitted to a queue, so no
        // work can still reference it. Order matches Vulkan's dependencies:
        // dependents first, then the image, then the memory backing it.
        unsafe {
            if self.view != vk::ImageView::null() {
                self.device.destroy_image_view(self.view, None);
            }
            if self.sampler != vk::Sampler::null() {
                self.device.destroy_sampler(self.sampler, None);
            }
            if let Some(conversion) = self.conversion {
                self.device
                    .destroy_sampler_ycbcr_conversion(conversion, None);
            }
            if self.image != vk::Image::null() {
                self.device.destroy_image(self.image, None);
            }
            if self.memory != vk::DeviceMemory::null() {
                self.device.free_memory(self.memory, None);
            }
        }
    }
}

fn memory_type_index(
    context: &VulkanContext,
    type_bits: u32,
    required: vk::MemoryPropertyFlags,
) -> Result<u32, AndroidPlatformError> {
    // SAFETY: the instance and physical device both belong to `context` and stay
    // valid for the borrow.
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
    // Imported `AHardwareBuffer` memory is not guaranteed to advertise the
    // preferred property flags, so fall back to any type the driver reported as
    // compatible with this buffer.
    for i in 0..mems.memory_type_count {
        if type_bits & (1 << i) != 0 {
            return Ok(i);
        }
    }
    Err(AndroidPlatformError::VulkanImportError(
        "no compatible memory type for AHardwareBuffer".into(),
    ))
}
