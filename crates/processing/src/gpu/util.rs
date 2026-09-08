use super::resources::{GpuFrameSlot, GpuImageProcessorInner};
use infers_core::{
    CoreError, DataType, ProcessingOptions, TensorShape,
};
use infers_gpu::GpuError;
use infers_gpu::{
    ash, ash::vk, gpu_allocator::MemoryLocation, AllocatedBuffer, VulkanContext, VulkanImage,
};
use std::sync::Arc;

/// Number of 32-bit words in the shader's `ProcessingOptions` push constant block.
const OPTIONS_WORDS: usize = 13;

/// Byte size of the shader's `ProcessingOptions` push constant block.
pub(crate) const PUSH_CONSTANT_BYTES: usize = std::mem::size_of::<ProcessingOptions>();

pub(crate) fn write_options_bytes(options: &ProcessingOptions) -> [u8; PUSH_CONSTANT_BYTES] {
    const _: () = assert!(
        PUSH_CONSTANT_BYTES == OPTIONS_WORDS * 4,
        "ProcessingOptions no longer matches the shader push constant layout"
    );

    let words: [u32; OPTIONS_WORDS] = [
        options.src_w,
        options.src_h,
        options.crop_x,
        options.crop_y,
        options.crop_w,
        options.crop_h,
        options.dest_w,
        options.dest_h,
        options.src_format as u32,
        options.dest_format as u32,
        options.fit_mode as u32,
        options.rotation_degrees.to_bits(),
        options.dest_layout as u32,
    ];

    let mut out = [0u8; PUSH_CONSTANT_BYTES];
    for (slot, word) in out.chunks_exact_mut(4).zip(words) {
        slot.copy_from_slice(&word.to_le_bytes());
    }
    out
}

pub(crate) fn spirv_words(bytes: &[u8]) -> Result<Vec<u32>, GpuError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(GpuError::Other("SPIR-V binary is not 4-byte aligned".into()));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

pub(crate) fn storage_binding(
    binding: u32,
    ty: vk::DescriptorType,
) -> vk::DescriptorSetLayoutBinding<'static> {
    vk::DescriptorSetLayoutBinding::default()
        .binding(binding)
        .descriptor_type(ty)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
}

pub(crate) fn buffer_write(
    set: vk::DescriptorSet,
    binding: u32,
    ty: vk::DescriptorType,
    info: &vk::DescriptorBufferInfo,
) -> vk::WriteDescriptorSet<'_> {
    vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(ty)
        .buffer_info(std::slice::from_ref(info))
}

pub(crate) fn reset_and_record<F>(
    slot: &GpuFrameSlot,
    vkd: &ash::Device,
    record: F,
) -> Result<(), GpuError>
where
    F: FnOnce(&ash::Device, vk::CommandBuffer) -> Result<(), GpuError>,
{
    unsafe {
        vkd.reset_command_buffer(slot.cmd, vk::CommandBufferResetFlags::empty())?;
        vkd.begin_command_buffer(
            slot.cmd,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
    }
    record(vkd, slot.cmd)?;
    unsafe { vkd.end_command_buffer(slot.cmd)? };
    Ok(())
}

pub(crate) fn pool_buffer<'a>(
    ctx: &VulkanContext,
    slot: &'a mut Option<AllocatedBuffer>,
    size: u64,
    usage: vk::BufferUsageFlags,
    location: MemoryLocation,
    name: &str,
) -> Result<&'a mut AllocatedBuffer, GpuError> {
    let grow = match slot.as_ref() {
        Some(buf) => buf.size < size,
        None => true,
    };
    if grow {
        if let Some(old) = slot.take() {
            ctx.destroy_buffer(old);
        }
        *slot = Some(ctx.create_buffer(size, usage, location, name)?);
    }
    Ok(slot.as_mut().expect("buffer slot populated above"))
}

pub(crate) fn validate_process(
    inner: &GpuImageProcessorInner,
    input: &VulkanImage,
    options: &ProcessingOptions,
) -> Result<(), CoreError> {
    if !Arc::ptr_eq(&inner.context, input.context()) {
        return Err(CoreError::DeviceMismatch {
            expected: inner.vulkan.info().clone(),
            actual: input.context().device_info().clone(),
        });
    }

    let (dest_w, dest_h) = (options.dest_w, options.dest_h);
    if dest_w == 0 || dest_h == 0 {
        return Err(CoreError::InvalidImageBuffer(
            "Destination dimensions must be non-zero".into(),
        ));
    }
    if options.dest_format.is_yuv() {
        return Err(CoreError::InvalidImageBuffer(
            "GPU convert kernels emit RGB888 or RGBF32, not YUV".into(),
        ));
    }

    let src_format = input.format();
    if src_format.is_yuv()
        && (!input.width().is_multiple_of(2) || !input.height().is_multiple_of(2))
    {
        return Err(CoreError::InvalidImageBuffer(
            "YUV 4:2:0 input width and height must be even".into(),
        ));
    }

    let (_crop_x, _crop_y, crop_w, crop_h) = options.effective_crop();
    if crop_w == 0 || crop_h == 0 {
        return Err(CoreError::InvalidImageBuffer(
            "Effective crop width and height must be greater than zero".into(),
        ));
    }

    Ok(())
}

pub(crate) fn output_shape_dtype(
    options: &ProcessingOptions,
) -> Result<(TensorShape, DataType), CoreError> {
    let (dest_w, dest_h) = (options.dest_w, options.dest_h);
    let shape = match options.dest_layout {
        processing_core::TensorLayout::Nhwc => {
            TensorShape::new(vec![1, dest_h as usize, dest_w as usize, 3])?
        }
        processing_core::TensorLayout::Nchw => {
            TensorShape::new(vec![1, 3, dest_h as usize, dest_w as usize])?
        }
    };
    let dtype = match options.dest_format {
        infers_core::ImageFormat::Rgb888 => DataType::U8,
        infers_core::ImageFormat::Rgbf32 => DataType::F32,
        infers_core::ImageFormat::Nv12 | infers_core::ImageFormat::I420 => {
            return Err(CoreError::InvalidImageBuffer(
                "GPU convert kernels emit RGB888 or RGBF32, not YUV".into(),
            ));
        }
    };
    Ok((shape, dtype))
}
