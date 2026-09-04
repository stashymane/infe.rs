use super::SHADERS;
use infers_core::CoreError;
use infers_gpu::GpuError;
use infers_core::{
    DataType, HardwareImage, ImageFormat, MaterializeTarget, Pending, ProcessingOptions, Tensor,
    TensorShape,
};
use infers_gpu::{
    ash::vk,
    gpu_allocator::MemoryLocation,
    tensor_from_allocated, tensor_from_external, AllocatedBuffer, buffer_barrier, SharedYcbcrSampler,
    Vulkan, VulkanBufferHandle, VulkanContext, VulkanImage, VulkanSampledImage,
};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

/// Configuration for [`GpuImageProcessor`].
#[derive(Clone, Copy, Debug)]
pub struct GpuImageProcessorOptions {
    /// Number of in-flight frame slots to pool. Default `1` (sync path).
    /// Reserved for a future async API; slots are still waited on synchronously.
    pub in_flight_slots: u32,
}

impl Default for GpuImageProcessorOptions {
    fn default() -> Self {
        Self {
            in_flight_slots: 1,
        }
    }
}

/// Query ExecuTorch Vulkan input staging buffers by session input slot.
pub type SessionStagingQuery =
    Arc<dyn Fn(usize) -> Result<VulkanBufferHandle, CoreError> + Send + Sync>;

thread_local! {
    static SESSION_STAGING: std::cell::RefCell<Option<SessionStagingQuery>> =
        const { std::cell::RefCell::new(None) };
}

/// Install a thread-local staging query for [`MaterializeTarget::SessionInput`] commits.
pub fn set_session_staging_query(query: Option<SessionStagingQuery>) {
    SESSION_STAGING.with(|slot| *slot.borrow_mut() = query);
}

fn staging_for_materialize(explicit: Option<&SessionStagingQuery>) -> Option<SessionStagingQuery> {
    explicit
        .cloned()
        .or_else(|| SESSION_STAGING.with(|slot| slot.borrow().clone()))
}

#[derive(Clone)]
enum GpuProcessInput {
    LinearUploaded {
        src: vk::Buffer,
        size: u64,
    },
    LinearUnstaged {
        staging: vk::Buffer,
        gpu: vk::Buffer,
        size: u64,
    },
    Sampled(Arc<VulkanSampledImage>),
}

#[derive(Clone, Copy, Debug)]
struct SampledDispatchInfo {
    image: vk::Image,
    view: vk::ImageView,
    is_ycbcr: bool,
    acquire_from_external: bool,
    current_layout: vk::ImageLayout,
}

impl GpuProcessInput {
    fn from_image(input: &VulkanImage) -> Result<Self, CoreError> {
        match input {
            VulkanImage::Linear(buf) => {
                if buf.staging_dirty() {
                    Self::linear_unstaged(buf)
                } else {
                    let gpu = buf.gpu_buffer().ok_or_else(|| {
                        CoreError::InvalidImageBuffer("linear GPU image buffer destroyed".into())
                    })?;
                    Ok(Self::LinearUploaded {
                        src: gpu.buffer,
                        size: gpu.size,
                    })
                }
            }
            VulkanImage::Sampled(sampled) => Ok(Self::Sampled(Arc::clone(sampled))),
        }
    }

    fn linear_unstaged(buf: &infers_gpu::VulkanImageBuffer) -> Result<Self, CoreError> {
        let staging = buf.staging_buffer().ok_or_else(|| {
            CoreError::InvalidImageBuffer("image staging buffer destroyed".into())
        })?;
        let gpu = buf.gpu_buffer().ok_or_else(|| {
            CoreError::InvalidImageBuffer("image gpu buffer destroyed".into())
        })?;
        Ok(Self::LinearUnstaged {
            staging: staging.buffer,
            gpu: gpu.buffer,
            size: gpu.size,
        })
    }
}

impl From<&Arc<VulkanSampledImage>> for SampledDispatchInfo {
    fn from(sampled: &Arc<VulkanSampledImage>) -> Self {
        Self {
            image: sampled.image(),
            view: sampled.view(),
            is_ycbcr: sampled.is_ycbcr(),
            acquire_from_external: sampled.acquire_from_external(),
            current_layout: sampled.current_layout(),
        }
    }
}

enum OutputDestination {
    Pooled,
    External(VulkanBufferHandle),
}

struct YcbcrPipeline {
    /// Keeps the immutable sampler alive for `set_layout`.
    _ycbcr: Arc<SharedYcbcrSampler>,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    /// One pre-allocated descriptor set per frame slot.
    desc_sets: Vec<vk::DescriptorSet>,
}

struct SampledPipelineBinding {
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    desc_set: vk::DescriptorSet,
}

struct GpuFrameSlot {
    dst: Option<AllocatedBuffer>,
    cmd: vk::CommandBuffer,
    desc_ssbo: vk::DescriptorSet,
    desc_image: vk::DescriptorSet,
    fence: vk::Fence,
}

struct DispatchResources {
    slots: Vec<GpuFrameSlot>,
    slot_index: usize,
    ycbcr_pipelines: HashMap<u64, YcbcrPipeline>,
}

struct GpuImageProcessorInner {
    vulkan: Vulkan,
    context: Arc<VulkanContext>,
    shader_module: vk::ShaderModule,
    pipeline_cache: vk::PipelineCache,
    set_layout_ssbo: vk::DescriptorSetLayout,
    set_layout_image: vk::DescriptorSetLayout,
    layout_ssbo: vk::PipelineLayout,
    layout_image: vk::PipelineLayout,
    pipeline_ssbo: vk::Pipeline,
    pipeline_image: vk::Pipeline,
    sampler: vk::Sampler,
    cmd_pool: vk::CommandPool,
    desc_pool: vk::DescriptorPool,
    record_lock: Mutex<DispatchResources>,
    /// Reused SSBO upload buffer for hardware-sourced frames (width, height, format, buffer).
    upload_buffer: Mutex<Option<(u32, u32, ImageFormat, infers_gpu::VulkanImageBuffer)>>,
}

/// GPU image processor that dispatches SPIR-V `convert_main` / `convert_image` on a shared context.
#[derive(Clone)]
pub struct GpuImageProcessor {
    inner: Arc<GpuImageProcessorInner>,
}

impl GpuImageProcessor {
    pub fn new(vulkan: Vulkan) -> Result<Self, CoreError> {
        Self::new_with_options(vulkan, GpuImageProcessorOptions::default())
    }

    pub fn new_with_options(
        vulkan: Vulkan,
        options: GpuImageProcessorOptions,
    ) -> Result<Self, CoreError> {
        Self::try_new_with_options(vulkan, options)
            .map(|inner| Self { inner: Arc::new(inner) })
            .map_err(CoreError::from)
    }

    fn try_new_with_options(
        vulkan: Vulkan,
        options: GpuImageProcessorOptions,
    ) -> Result<GpuImageProcessorInner, GpuError> {
        let slot_count = options.in_flight_slots.max(1) as usize;
        let context = Arc::clone(vulkan.context());
        let vkd = context.device();

        let words = spirv_words(SHADERS)?;
        let shader_module = unsafe {
            vkd.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)?
        };

        let pipeline_cache = unsafe {
            vkd.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)
        }?;

        let set_layout_ssbo = unsafe {
            vkd.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                    storage_binding(0, vk::DescriptorType::STORAGE_BUFFER),
                    storage_binding(1, vk::DescriptorType::STORAGE_BUFFER),
                ]),
                None,
            )?
        };
        let set_layout_image = unsafe {
            vkd.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE),
                    storage_binding(1, vk::DescriptorType::STORAGE_BUFFER),
                ]),
                None,
            )?
        };

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(PUSH_CONSTANT_BYTES as u32);
        let layout_ssbo = unsafe {
            vkd.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[set_layout_ssbo])
                    .push_constant_ranges(&[push_constant_range]),
                None,
            )?
        };
        let layout_image = unsafe {
            vkd.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[set_layout_image])
                    .push_constant_ranges(&[push_constant_range]),
                None,
            )?
        };

        let stage_main = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(c"convert_main");
        let stage_image = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(c"convert_image");
        let infos = [
            vk::ComputePipelineCreateInfo::default()
                .stage(stage_main)
                .layout(layout_ssbo),
            vk::ComputePipelineCreateInfo::default()
                .stage(stage_image)
                .layout(layout_image),
        ];
        let pipelines = unsafe {
            vkd.create_compute_pipelines(pipeline_cache, &infos, None)
                .map_err(|(_, err)| err)?
        };
        let pipeline_ssbo = pipelines[0];
        let pipeline_image = pipelines[1];

        let sampler = unsafe {
            vkd.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )?
        };

        let cmd_pool = unsafe {
            vkd.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(context.queue_family_index())
                    .flags(
                        vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER
                            | vk::CommandPoolCreateFlags::TRANSIENT,
                    ),
                None,
            )?
        };

        let desc_pool = unsafe {
            vkd.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    // Base slots (ssbo+image) plus a few YCbCr format variants.
                    .max_sets((slot_count * 2 + slot_count * 4 + 8) as u32)
                    .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                    .pool_sizes(&[
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: (slot_count * 4 + slot_count * 4 + 16) as u32,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: (slot_count * 2 + slot_count * 4 + 8) as u32,
                        },
                    ]),
                None,
            )?
        };

        let mut slots = Vec::with_capacity(slot_count);
        for _ in 0..slot_count {
            slots.push(create_frame_slot(
                vkd,
                cmd_pool,
                desc_pool,
                set_layout_ssbo,
                set_layout_image,
            )?);
        }

        Ok(GpuImageProcessorInner {
            vulkan,
            context,
            shader_module,
            pipeline_cache,
            set_layout_ssbo,
            set_layout_image,
            layout_ssbo,
            layout_image,
            pipeline_ssbo,
            pipeline_image,
            sampler,
            cmd_pool,
            desc_pool,
            record_lock: Mutex::new(DispatchResources {
                slots,
                slot_index: 0,
                ycbcr_pipelines: HashMap::new(),
            }),
            upload_buffer: Mutex::new(None),
        })
    }

    pub fn vulkan(&self) -> &Vulkan {
        &self.inner.vulkan
    }

    pub fn context(&self) -> &Arc<VulkanContext> {
        &self.inner.context
    }

    pub fn process(
        &self,
        input: &VulkanImage,
        options: &ProcessingOptions,
    ) -> Result<Pending<Vulkan>, CoreError> {
        self.process_with_staging(input, options, None)
    }

    pub(crate) fn process_with_staging(
        &self,
        input: &VulkanImage,
        options: &ProcessingOptions,
        session_staging: Option<SessionStagingQuery>,
    ) -> Result<Pending<Vulkan>, CoreError> {
        validate_process(&self.inner, input, options)?;
        let (shape, dtype) = output_shape_dtype(options)?;
        let process_input = GpuProcessInput::from_image(input)?;
        let mut kernel_opts = *options;
        kernel_opts.src_w = input.width();
        kernel_opts.src_h = input.height();
        kernel_opts.src_format = input.format();

        let inner = Arc::clone(&self.inner);
        let vulkan = inner.vulkan.clone();
        Ok(Pending::schedule(
            vulkan,
            shape,
            dtype,
            Box::new(move |target| {
                inner.materialize(
                    &kernel_opts,
                    process_input,
                    target,
                    session_staging.as_ref(),
                )
            }),
        ))
    }

    pub(crate) fn materialize_from_hardware(
        &self,
        device: &Vulkan,
        hardware: Arc<HardwareImage>,
        options: &ProcessingOptions,
        target: MaterializeTarget,
        session_staging: Option<&SessionStagingQuery>,
    ) -> Result<Tensor<Vulkan>, CoreError> {
        let width = hardware.width();
        let height = hardware.height();
        let format = hardware.format();
        let mut buffer = self
            .inner
            .take_upload_buffer(device, width, height, format)?;
        let result = (|| {
            buffer.write_staging(hardware.as_bytes())?;
            let process_input = GpuProcessInput::linear_unstaged(&buffer)?;

            let mut kernel_opts = *options;
            kernel_opts.src_w = width;
            kernel_opts.src_h = height;
            kernel_opts.src_format = format;

            self.inner.materialize(
                &kernel_opts,
                process_input,
                target,
                session_staging,
            )
        })();
        self.inner.store_upload_buffer(width, height, format, buffer);
        result
    }
}

impl Drop for GpuImageProcessor {
    fn drop(&mut self) {
        if Arc::strong_count(&self.inner) != 1 {
            return;
        }
        let inner = Arc::get_mut(&mut self.inner).expect("unique Arc in Drop");
        drop_gpu_processor_inner(inner);
    }
}

impl GpuImageProcessorInner {
    fn take_upload_buffer(
        &self,
        device: &Vulkan,
        width: u32,
        height: u32,
        format: ImageFormat,
    ) -> Result<infers_gpu::VulkanImageBuffer, CoreError> {
        let mut slot = self.upload_buffer.lock();
        if let Some((w, h, fmt, buffer)) = slot.take() {
            if w == width && h == height && fmt == format {
                return Ok(buffer);
            }
        }
        device.image_buffer(width, height, format)
    }

    fn store_upload_buffer(
        &self,
        width: u32,
        height: u32,
        format: ImageFormat,
        buffer: infers_gpu::VulkanImageBuffer,
    ) {
        *self.upload_buffer.lock() = Some((width, height, format, buffer));
    }

    fn materialize(
        &self,
        options: &ProcessingOptions,
        input: GpuProcessInput,
        target: MaterializeTarget,
        session_staging: Option<&SessionStagingQuery>,
    ) -> Result<Tensor<Vulkan>, CoreError> {
        let staging = staging_for_materialize(session_staging);
        let (shape, dtype) = output_shape_dtype(options)?;
        let dst_size = shape.byte_size(dtype) as u64;

        match input {
            GpuProcessInput::LinearUploaded { src, size } => self
                .dispatch_linear_uploaded(
                    src,
                    size,
                    options,
                    shape,
                    dtype,
                    dst_size,
                    target,
                    staging.as_ref(),
                )
                .map_err(CoreError::from),
            GpuProcessInput::LinearUnstaged {
                staging: staging_buf,
                gpu,
                size,
            } => self
                .dispatch_linear_unstaged(
                    staging_buf,
                    gpu,
                    size,
                    options,
                    shape,
                    dtype,
                    dst_size,
                    target,
                    staging.as_ref(),
                )
                .map_err(CoreError::from),
            GpuProcessInput::Sampled(sampled) => self
                .dispatch_sampled(
                    &sampled,
                    options,
                    shape,
                    dtype,
                    dst_size,
                    target,
                    staging.as_ref(),
                )
                .map_err(CoreError::from),
        }
    }

    fn dispatch_linear_uploaded(
        &self,
        src_gpu: vk::Buffer,
        src_size: u64,
        options: &ProcessingOptions,
        shape: TensorShape,
        dtype: DataType,
        dst_size: u64,
        target: MaterializeTarget,
        session_staging: Option<&SessionStagingQuery>,
    ) -> Result<Tensor<Vulkan>, GpuError> {
        let ctx = self.context.as_ref();
        let mut dispatch = self.record_lock.lock();
        let slot = current_slot(&mut dispatch);

        let options_bytes = write_options_bytes(options);

        let (dst_buffer, dst_info, destination) = resolve_materialize_target(
            ctx,
            &mut slot.dst,
            dst_size,
            target,
            session_staging,
        )
        .map_err(|err| GpuError::Other(err.to_string()))?;

        let vkd = ctx.device();
        let src_info = vk::DescriptorBufferInfo::default()
            .buffer(src_gpu)
            .offset(0)
            .range(src_size);
        let writes = [
            buffer_write(
                slot.desc_ssbo,
                0,
                vk::DescriptorType::STORAGE_BUFFER,
                &src_info,
            ),
            buffer_write(
                slot.desc_ssbo,
                1,
                vk::DescriptorType::STORAGE_BUFFER,
                &dst_info,
            ),
        ];
        unsafe { vkd.update_descriptor_sets(&writes, &[]) };

        reset_and_record(slot, vkd, |vkd, cmd| {
            unsafe {
                buffer_barrier(
                    vkd,
                    cmd,
                    src_gpu,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::SHADER_READ,
                );
                vkd.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline_ssbo);
                vkd.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.layout_ssbo,
                    0,
                    &[slot.desc_ssbo],
                    &[],
                );
                vkd.cmd_push_constants(
                    cmd,
                    self.layout_ssbo,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &options_bytes,
                );
                vkd.cmd_dispatch(
                    cmd,
                    options.dest_w.div_ceil(16),
                    options.dest_h.div_ceil(16),
                    1,
                );
                buffer_barrier(
                    vkd,
                    cmd,
                    dst_buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::AccessFlags::SHADER_WRITE,
                    vk::AccessFlags::TRANSFER_READ,
                );
            }
            Ok(())
        })?;

        self.submit_and_wait(slot)?;
        finish_materialized_tensor(
            &self.vulkan,
            shape,
            dtype,
            destination,
            slot.dst.take(),
        )
        .and_then(|tensor| {
            advance_slot(&mut dispatch);
            Ok(tensor)
        })
    }

    fn dispatch_linear_unstaged(
        &self,
        staging: vk::Buffer,
        gpu: vk::Buffer,
        size: u64,
        options: &ProcessingOptions,
        shape: TensorShape,
        dtype: DataType,
        dst_size: u64,
        target: MaterializeTarget,
        session_staging: Option<&SessionStagingQuery>,
    ) -> Result<Tensor<Vulkan>, GpuError> {
        let ctx = self.context.as_ref();
        let mut dispatch = self.record_lock.lock();
        let slot = current_slot(&mut dispatch);

        let options_bytes = write_options_bytes(options);

        let (dst_buffer, dst_info, destination) = resolve_materialize_target(
            ctx,
            &mut slot.dst,
            dst_size,
            target,
            session_staging,
        )
        .map_err(|err| GpuError::Other(err.to_string()))?;

        let vkd = ctx.device();
        let src_info = vk::DescriptorBufferInfo::default()
            .buffer(gpu)
            .offset(0)
            .range(size);
        let writes = [
            buffer_write(
                slot.desc_ssbo,
                0,
                vk::DescriptorType::STORAGE_BUFFER,
                &src_info,
            ),
            buffer_write(
                slot.desc_ssbo,
                1,
                vk::DescriptorType::STORAGE_BUFFER,
                &dst_info,
            ),
        ];
        unsafe { vkd.update_descriptor_sets(&writes, &[]) };

        reset_and_record(slot, vkd, |vkd, cmd| {
            unsafe {
                vkd.cmd_copy_buffer(
                    cmd,
                    staging,
                    gpu,
                    &[vk::BufferCopy::default().size(size)],
                );
                buffer_barrier(
                    vkd,
                    cmd,
                    gpu,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::SHADER_READ,
                );
                vkd.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline_ssbo);
                vkd.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.layout_ssbo,
                    0,
                    &[slot.desc_ssbo],
                    &[],
                );
                vkd.cmd_push_constants(
                    cmd,
                    self.layout_ssbo,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &options_bytes,
                );
                vkd.cmd_dispatch(
                    cmd,
                    options.dest_w.div_ceil(16),
                    options.dest_h.div_ceil(16),
                    1,
                );
                buffer_barrier(
                    vkd,
                    cmd,
                    dst_buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::AccessFlags::SHADER_WRITE,
                    vk::AccessFlags::TRANSFER_READ,
                );
            }
            Ok(())
        })?;

        self.submit_and_wait(slot)?;
        finish_materialized_tensor(
            &self.vulkan,
            shape,
            dtype,
            destination,
            slot.dst.take(),
        )
        .and_then(|tensor| {
            advance_slot(&mut dispatch);
            Ok(tensor)
        })
    }

    fn dispatch_sampled(
        &self,
        sampled: &Arc<VulkanSampledImage>,
        options: &ProcessingOptions,
        shape: TensorShape,
        dtype: DataType,
        dst_size: u64,
        target: MaterializeTarget,
        session_staging: Option<&SessionStagingQuery>,
    ) -> Result<Tensor<Vulkan>, GpuError> {
        let ctx = self.context.as_ref();
        let mut dispatch = self.record_lock.lock();
        let slot_index = dispatch.slot_index;
        let dispatch_info = SampledDispatchInfo::from(sampled);
        let binding = self.resolve_sampled_binding(&mut dispatch, sampled, slot_index)?;
        let slot = &mut dispatch.slots[slot_index];

        let options_bytes = write_options_bytes(options);

        let (dst_buffer, dst_info, destination) = resolve_materialize_target(
            ctx,
            &mut slot.dst,
            dst_size,
            target,
            session_staging,
        )
        .map_err(|err| GpuError::Other(err.to_string()))?;

        let vkd = ctx.device();
        let image_info = vk::DescriptorImageInfo::default()
            .sampler(if dispatch_info.is_ycbcr {
                vk::Sampler::null()
            } else {
                self.sampler
            })
            .image_view(dispatch_info.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(binding.desc_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&image_info)),
            buffer_write(
                binding.desc_set,
                1,
                vk::DescriptorType::STORAGE_BUFFER,
                &dst_info,
            ),
        ];
        unsafe { vkd.update_descriptor_sets(&writes, &[]) };

        reset_and_record(slot, vkd, |vkd, cmd| {
            let target_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            if dispatch_info.current_layout != target_layout {
                let src_queue = if dispatch_info.acquire_from_external {
                    vk::QUEUE_FAMILY_EXTERNAL
                } else {
                    vk::QUEUE_FAMILY_IGNORED
                };
                let barrier = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(dispatch_info.current_layout)
                    .new_layout(target_layout)
                    .src_queue_family_index(src_queue)
                    .dst_queue_family_index(if dispatch_info.acquire_from_external {
                        ctx.queue_family_index()
                    } else {
                        vk::QUEUE_FAMILY_IGNORED
                    })
                    .image(dispatch_info.image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                unsafe {
                    vkd.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[barrier],
                    );
                }
                sampled.set_layout(target_layout);
            }
            unsafe {
                vkd.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, binding.pipeline);
                vkd.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    binding.pipeline_layout,
                    0,
                    &[binding.desc_set],
                    &[],
                );
                vkd.cmd_push_constants(
                    cmd,
                    binding.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &options_bytes,
                );
                vkd.cmd_dispatch(
                    cmd,
                    options.dest_w.div_ceil(16),
                    options.dest_h.div_ceil(16),
                    1,
                );
                buffer_barrier(
                    vkd,
                    cmd,
                    dst_buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::AccessFlags::SHADER_WRITE,
                    vk::AccessFlags::TRANSFER_READ,
                );
            }
            Ok(())
        })?;

        self.submit_and_wait(slot)?;
        finish_materialized_tensor(
            &self.vulkan,
            shape,
            dtype,
            destination,
            slot.dst.take(),
        )
        .and_then(|tensor| {
            advance_slot(&mut dispatch);
            Ok(tensor)
        })
    }

    fn resolve_sampled_binding(
        &self,
        dispatch: &mut DispatchResources,
        sampled: &Arc<VulkanSampledImage>,
        slot_index: usize,
    ) -> Result<SampledPipelineBinding, GpuError> {
        if !sampled.is_ycbcr() {
            return Ok(SampledPipelineBinding {
                pipeline_layout: self.layout_image,
                pipeline: self.pipeline_image,
                desc_set: dispatch.slots[slot_index].desc_image,
            });
        }

        let ycbcr = sampled.shared_ycbcr().ok_or_else(|| {
            GpuError::Other("YCbCr sampled image missing shared sampler".into())
        })?;
        let ctx = self.context.as_ref();
        let vkd = ctx.device();
        let key = ycbcr.sampler_key();
        let slot_count = dispatch.slots.len();

        if let Some(cached) = dispatch.ycbcr_pipelines.get_mut(&key) {
            if cached.desc_sets.len() != slot_count {
                unsafe {
                    let _ = vkd.free_descriptor_sets(self.desc_pool, &cached.desc_sets);
                }
                cached.desc_sets = allocate_ycbcr_desc_sets(
                    vkd,
                    self.desc_pool,
                    cached.set_layout,
                    slot_count,
                )?;
            }
            return Ok(SampledPipelineBinding {
                pipeline_layout: cached.layout,
                pipeline: cached.pipeline,
                desc_set: cached.desc_sets[slot_index],
            });
        }

        let immutable = [ycbcr.sampler()];
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .immutable_samplers(&immutable),
            storage_binding(1, vk::DescriptorType::STORAGE_BUFFER),
        ];
        let set_layout = unsafe {
            vkd.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?
        };
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(PUSH_CONSTANT_BYTES as u32);
        let pipeline_layout = unsafe {
            vkd.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[set_layout])
                    .push_constant_ranges(&[push_constant_range]),
                None,
            )?
        };
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(self.shader_module)
            .name(c"convert_image");
        let pipeline = unsafe {
            vkd.create_compute_pipelines(
                self.pipeline_cache,
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(pipeline_layout)],
                None,
            )
            .map_err(|(_, err)| err)?[0]
        };
        let desc_sets =
            allocate_ycbcr_desc_sets(vkd, self.desc_pool, set_layout, slot_count)?;
        let desc_set = desc_sets[slot_index];
        dispatch.ycbcr_pipelines.insert(
            key,
            YcbcrPipeline {
                _ycbcr: Arc::clone(ycbcr),
                set_layout,
                layout: pipeline_layout,
                pipeline,
                desc_sets,
            },
        );
        Ok(SampledPipelineBinding {
            pipeline_layout,
            pipeline,
            desc_set,
        })
    }

    fn submit_and_wait(&self, slot: &GpuFrameSlot) -> Result<(), GpuError> {
        let vkd = self.context.device();
        unsafe { vkd.reset_fences(&[slot.fence])? };
        self.context
            .submit(slot.cmd, slot.fence)
            .and_then(|()| self.context.wait_fence(slot.fence))
    }
}

fn drop_gpu_processor_inner(inner: &mut GpuImageProcessorInner) {
    let vkd = inner.context.device();
    let _ = unsafe { vkd.device_wait_idle() };
    let mut dispatch = inner.record_lock.lock();
    let ctx = inner.context.as_ref();
    for slot in &mut dispatch.slots {
        if let Some(buf) = slot.dst.take() {
            ctx.destroy_buffer(buf);
        }
        unsafe {
            vkd.destroy_fence(slot.fence, None);
            vkd.free_command_buffers(inner.cmd_pool, &[slot.cmd]);
        }
    }
    unsafe {
        for cached in dispatch.ycbcr_pipelines.values() {
            vkd.destroy_pipeline(cached.pipeline, None);
            vkd.destroy_pipeline_layout(cached.layout, None);
            vkd.destroy_descriptor_set_layout(cached.set_layout, None);
        }
        vkd.destroy_pipeline(inner.pipeline_ssbo, None);
        vkd.destroy_pipeline(inner.pipeline_image, None);
        vkd.destroy_pipeline_layout(inner.layout_ssbo, None);
        vkd.destroy_pipeline_layout(inner.layout_image, None);
        vkd.destroy_descriptor_set_layout(inner.set_layout_ssbo, None);
        vkd.destroy_descriptor_set_layout(inner.set_layout_image, None);
        vkd.destroy_pipeline_cache(inner.pipeline_cache, None);
        vkd.destroy_shader_module(inner.shader_module, None);
        vkd.destroy_sampler(inner.sampler, None);
        vkd.destroy_command_pool(inner.cmd_pool, None);
        vkd.destroy_descriptor_pool(inner.desc_pool, None);
    }
}

fn validate_process(
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

fn resolve_materialize_target(
    ctx: &VulkanContext,
    dst_slot: &mut Option<AllocatedBuffer>,
    dst_size: u64,
    target: MaterializeTarget,
    session_staging: Option<&SessionStagingQuery>,
) -> Result<(vk::Buffer, vk::DescriptorBufferInfo, OutputDestination), CoreError> {
    match target {
        MaterializeTarget::Owned => {
            pool_buffer(
                ctx,
                dst_slot,
                dst_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::GpuOnly,
                "dst-ssbo",
            )
            .map_err(CoreError::from)?;
            let dst = dst_slot.as_ref().expect("dst pooled");
            Ok((
                dst.buffer,
                vk::DescriptorBufferInfo::default()
                    .buffer(dst.buffer)
                    .offset(0)
                    .range(dst.size),
                OutputDestination::Pooled,
            ))
        }
        MaterializeTarget::SessionInput { slot } => {
            let query = session_staging.ok_or_else(|| {
                CoreError::BufferTransferFailed(
                    "session input staging query not configured for GPU preprocess".into(),
                )
            })?;
            let handle = query(slot)?;
            if handle.size < dst_size {
                return Err(CoreError::BufferTransferFailed(format!(
                    "session input staging buffer needs {dst_size} bytes but only {} are available",
                    handle.size
                )));
            }
            Ok((
                handle.buffer,
                vk::DescriptorBufferInfo::default()
                    .buffer(handle.buffer)
                    .offset(handle.offset)
                    .range(dst_size),
                OutputDestination::External(handle),
            ))
        }
    }
}

fn finish_materialized_tensor(
    vulkan: &Vulkan,
    shape: TensorShape,
    dtype: DataType,
    destination: OutputDestination,
    pooled_dst: Option<AllocatedBuffer>,
) -> Result<Tensor<Vulkan>, GpuError> {
    match destination {
        OutputDestination::Pooled => {
            let buffer = pooled_dst.ok_or_else(|| {
                GpuError::Other("pooled output buffer missing after dispatch".into())
            })?;
            Ok(tensor_from_allocated(vulkan, shape, dtype, buffer))
        }
        OutputDestination::External(handle) => Ok(tensor_from_external(
            vulkan,
            shape,
            dtype,
            handle,
        )),
    }
}

fn current_slot<'a>(dispatch: &'a mut DispatchResources) -> &'a mut GpuFrameSlot {
    &mut dispatch.slots[dispatch.slot_index]
}

fn advance_slot(dispatch: &mut DispatchResources) {
    dispatch.slot_index = (dispatch.slot_index + 1) % dispatch.slots.len();
}

fn create_frame_slot(
    vkd: &ash::Device,
    cmd_pool: vk::CommandPool,
    desc_pool: vk::DescriptorPool,
    set_layout_ssbo: vk::DescriptorSetLayout,
    set_layout_image: vk::DescriptorSetLayout,
) -> Result<GpuFrameSlot, GpuError> {
    let fence = unsafe { vkd.create_fence(&vk::FenceCreateInfo::default(), None) }?;
    let cmd = unsafe {
        vkd.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(cmd_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )?
    }[0];
    let layouts_ssbo = [set_layout_ssbo];
    let layouts_image = [set_layout_image];
    let desc_ssbo = unsafe {
        vkd.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(desc_pool)
                .set_layouts(&layouts_ssbo),
        )?
    }[0];
    let desc_image = unsafe {
        vkd.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(desc_pool)
                .set_layouts(&layouts_image),
        )?
    }[0];
    Ok(GpuFrameSlot {
        dst: None,
        cmd,
        desc_ssbo,
        desc_image,
        fence,
    })
}

fn allocate_ycbcr_desc_sets(
    vkd: &ash::Device,
    desc_pool: vk::DescriptorPool,
    set_layout: vk::DescriptorSetLayout,
    count: usize,
) -> Result<Vec<vk::DescriptorSet>, GpuError> {
    let layouts: Vec<_> = (0..count).map(|_| set_layout).collect();
    Ok(unsafe {
        vkd.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(desc_pool)
                .set_layouts(&layouts),
        )?
    })
}

fn output_shape_dtype(options: &ProcessingOptions) -> Result<(TensorShape, DataType), CoreError> {
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
        ImageFormat::Rgb888 => DataType::U8,
        ImageFormat::Rgbf32 => DataType::F32,
        ImageFormat::Nv12 | ImageFormat::I420 => {
            return Err(CoreError::InvalidImageBuffer(
                "GPU convert kernels emit RGB888 or RGBF32, not YUV".into(),
            ));
        }
    };
    Ok((shape, dtype))
}

fn reset_and_record<F>(
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
    unsafe { vkd.end_command_buffer(slot.cmd) }?;
    Ok(())
}

fn pool_buffer<'a>(
    ctx: &VulkanContext,
    slot: &'a mut Option<AllocatedBuffer>,
    size: u64,
    usage: vk::BufferUsageFlags,
    location: MemoryLocation,
    name: &str,
) -> Result<&'a mut AllocatedBuffer, GpuError> {
    let grow = slot.as_ref().is_none_or(|buf| buf.size < size);
    if grow {
        if let Some(old) = slot.take() {
            ctx.destroy_buffer(old);
        }
        *slot = Some(ctx.create_buffer(size, usage, location, name)?);
    }
    Ok(slot.as_mut().expect("buffer slot populated above"))
}

fn storage_binding(binding: u32, ty: vk::DescriptorType) -> vk::DescriptorSetLayoutBinding<'static> {
    vk::DescriptorSetLayoutBinding::default()
        .binding(binding)
        .descriptor_type(ty)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
}

fn buffer_write(
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

fn spirv_words(bytes: &[u8]) -> Result<Vec<u32>, GpuError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(GpuError::Other("SPIR-V binary is not 4-byte aligned".into()));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Number of 32-bit words in the shader's `ProcessingOptions` push constant block.
const OPTIONS_WORDS: usize = 13;

/// Byte size of the shader's `ProcessingOptions` push constant block.
const PUSH_CONSTANT_BYTES: usize = std::mem::size_of::<ProcessingOptions>();

fn write_options_bytes(options: &ProcessingOptions) -> [u8; PUSH_CONSTANT_BYTES] {
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
        options.rotation as u32,
        options.dest_layout as u32,
    ];

    let mut out = [0u8; PUSH_CONSTANT_BYTES];
    for (slot, word) in out.chunks_exact_mut(4).zip(words) {
        slot.copy_from_slice(&word.to_le_bytes());
    }
    out
}
