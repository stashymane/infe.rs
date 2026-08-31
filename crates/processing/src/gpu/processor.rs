use super::SHADERS;
use super::buffer::GpuTensorBuffer;
use infers_core::CoreError;
use infers_gpu::GpuError;
use infers_core::{
    DataType, Device, ImageFormat, ImageInputBuffer, ProcessingOptions, TensorBuffer, TensorShape,
};
use infers_gpu::ash::vk;
use infers_gpu::ash::vk::Handle;
use infers_gpu::gpu_allocator::MemoryLocation;
use infers_gpu::{AllocatedBuffer, buffer_barrier, VulkanContext, VulkanSampledImage};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

struct YcbcrPipeline {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

struct FramePool {
    staging_src: Option<AllocatedBuffer>,
    src: Option<AllocatedBuffer>,
    ubo: Option<AllocatedBuffer>,
    fence: vk::Fence,
}

impl FramePool {
    fn new(vkd: &ash::Device) -> Result<Self, GpuError> {
        // SAFETY: default fence create info borrows nothing.
        let fence = unsafe { vkd.create_fence(&vk::FenceCreateInfo::default(), None) }?;
        Ok(Self {
            staging_src: None,
            src: None,
            ubo: None,
            fence,
        })
    }
}

struct DispatchResources {
    frame: FramePool,
    // YCbCr set layouts bind an immutable sampler; cache one pipeline per sampler handle.
    ycbcr_pipelines: HashMap<u64, YcbcrPipeline>,
}

/// GPU image processor that dispatches SPIR-V `convert_main` / `convert_image` on a shared context.
pub struct GpuImageProcessor {
    context: Arc<VulkanContext>,
    device: Device,
    shader_module: vk::ShaderModule,
    set_layout_ssbo: vk::DescriptorSetLayout,
    set_layout_image: vk::DescriptorSetLayout,
    layout_ssbo: vk::PipelineLayout,
    layout_image: vk::PipelineLayout,
    pipeline_ssbo: vk::Pipeline,
    pipeline_image: vk::Pipeline,
    sampler: vk::Sampler,
    cmd_pool: vk::CommandPool,
    desc_pool: Mutex<vk::DescriptorPool>,
    record_lock: Mutex<DispatchResources>,
}

impl GpuImageProcessor {
    pub fn new(context: Arc<VulkanContext>) -> Result<Self, CoreError> {
        Self::try_new(context).map_err(CoreError::from)
    }

    fn try_new(context: Arc<VulkanContext>) -> Result<Self, GpuError> {
        let device = context.logical_device().clone();
        let vkd = context.device();

        let words = spirv_words(SHADERS)?;
        let shader_module = unsafe {
            vkd.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)?
        };

        let set_layout_ssbo = unsafe {
            vkd.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                    storage_binding(0, vk::DescriptorType::UNIFORM_BUFFER),
                    storage_binding(1, vk::DescriptorType::STORAGE_BUFFER),
                    storage_binding(2, vk::DescriptorType::STORAGE_BUFFER),
                ]),
                None,
            )?
        };
        let set_layout_image = unsafe {
            vkd.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                    storage_binding(0, vk::DescriptorType::UNIFORM_BUFFER),
                    storage_binding(1, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
                    storage_binding(2, vk::DescriptorType::STORAGE_BUFFER),
                ]),
                None,
            )?
        };

        let layout_ssbo = unsafe {
            vkd.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout_ssbo]),
                None,
            )?
        };
        let layout_image = unsafe {
            vkd.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout_image]),
                None,
            )?
        };

        // C-string literals: no allocation and no fallible NUL check.
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
            vkd.create_compute_pipelines(vk::PipelineCache::null(), &infos, None)
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
                    .max_sets(8)
                    .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                    .pool_sizes(&[
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::UNIFORM_BUFFER,
                            descriptor_count: 8,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 16,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: 8,
                        },
                    ]),
                None,
            )?
        };

        let frame_pool = FramePool::new(vkd)?;

        Ok(Self {
            context,
            device,
            shader_module,
            set_layout_ssbo,
            set_layout_image,
            layout_ssbo,
            layout_image,
            pipeline_ssbo,
            pipeline_image,
            sampler,
            cmd_pool,
            desc_pool: Mutex::new(desc_pool),
            record_lock: Mutex::new(DispatchResources {
                frame: frame_pool,
                ycbcr_pipelines: HashMap::new(),
            }),
        })
    }

    pub fn context(&self) -> &Arc<VulkanContext> {
        &self.context
    }

    pub fn process(
        &self,
        input: &dyn ImageInputBuffer,
        options: &ProcessingOptions,
    ) -> Result<Box<dyn TensorBuffer>, CoreError> {
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

        let mut kernel_opts = *options;
        kernel_opts.src_w = input.width();
        kernel_opts.src_h = input.height();
        kernel_opts.src_format = src_format;

        if let Some(bytes) = input.as_bytes() {
            let expected = src_format.frame_bytes(input.width(), input.height()) as usize;
            if bytes.len() != expected {
                return Err(CoreError::InvalidImageBuffer(format!(
                    "YUV/RGB source size mismatch: expected {expected} bytes, got {}",
                    bytes.len()
                )));
            }
            return self
                .dispatch_ssbo(bytes, &kernel_opts, shape, dtype)
                .map_err(CoreError::from);
        }

        if let Some(sampled) = input.as_any().downcast_ref::<VulkanSampledImage>() {
            return self
                .dispatch_sampled(sampled, &kernel_opts, shape, dtype)
                .map_err(CoreError::from);
        }

        Err(CoreError::InvalidImageBuffer(
            "GPU sampled path requires host bytes or a Vulkan sampled image".into(),
        ))
    }

    fn dispatch_ssbo(
        &self,
        src_bytes: &[u8],
        options: &ProcessingOptions,
        shape: TensorShape,
        dtype: DataType,
    ) -> Result<Box<dyn TensorBuffer>, GpuError> {
        let dst_size = shape.byte_size(dtype) as u64;
        let ctx = self.context.as_ref();
        let src_size = src_bytes.len() as u64;

        let mut dispatch = self.record_lock.lock();
        pool_buffer(
            ctx,
            &mut dispatch.frame.staging_src,
            src_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            "src-staging",
        )?;
        pool_buffer(
            ctx,
            &mut dispatch.frame.src,
            src_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuOnly,
            "src-ssbo",
        )?;
        let ubo_bytes = options_bytes(options);
        pool_buffer(
            ctx,
            &mut dispatch.frame.ubo,
            ubo_bytes.len() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
            "convert-ubo",
        )?;
        VulkanContext::write_allocation(
            &mut dispatch.frame.staging_src.as_mut().unwrap().allocation,
            src_bytes,
        )?;
        VulkanContext::write_allocation(
            &mut dispatch.frame.ubo.as_mut().unwrap().allocation,
            &ubo_bytes,
        )?;

        let staging_src_buf = dispatch.frame.staging_src.as_ref().unwrap().buffer;
        let src_buf = dispatch.frame.src.as_ref().unwrap().buffer;
        let src_buf_size = dispatch.frame.src.as_ref().unwrap().size;
        let ubo_buf = dispatch.frame.ubo.as_ref().unwrap().buffer;

        let dst = ctx.create_buffer(
            dst_size,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuOnly,
            "dst-ssbo",
        )?;

        let vkd = ctx.device();
        let desc_pool = *self.desc_pool.lock();
        // SAFETY: the pool belongs to this device and `record_lock` is held, so
        // no other dispatch holds sets allocated from it.
        unsafe { vkd.reset_descriptor_pool(desc_pool, vk::DescriptorPoolResetFlags::empty())? };
        let set = unsafe {
            vkd.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(desc_pool)
                    .set_layouts(&[self.set_layout_ssbo]),
            )?
        }[0];

        let ubo_info = vk::DescriptorBufferInfo::default()
            .buffer(ubo_buf)
            .offset(0)
            .range(ubo_bytes.len() as u64);
        let src_info = vk::DescriptorBufferInfo::default()
            .buffer(src_buf)
            .offset(0)
            .range(src_buf_size);
        let dst_info = vk::DescriptorBufferInfo::default()
            .buffer(dst.buffer)
            .offset(0)
            .range(dst.size);
        let writes = [
            buffer_write(set, 0, vk::DescriptorType::UNIFORM_BUFFER, &ubo_info),
            buffer_write(set, 1, vk::DescriptorType::STORAGE_BUFFER, &src_info),
            buffer_write(set, 2, vk::DescriptorType::STORAGE_BUFFER, &dst_info),
        ];
        unsafe { vkd.update_descriptor_sets(&writes, &[]) };

        let cmd = self.alloc_cmd()?;
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { vkd.begin_command_buffer(cmd, &begin)? };
        let copy = vk::BufferCopy::default().size(src_size);
        unsafe {
            vkd.cmd_copy_buffer(cmd, staging_src_buf, src_buf, &[copy]);
            buffer_barrier(
                vkd,
                cmd,
                src_buf,
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
                &[set],
                &[],
            );
            let gx = options.dest_w.div_ceil(16);
            let gy = options.dest_h.div_ceil(16);
            vkd.cmd_dispatch(cmd, gx, gy, 1);
            buffer_barrier(
                vkd,
                cmd,
                dst.buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::SHADER_WRITE,
                vk::AccessFlags::TRANSFER_READ,
            );
            vkd.end_command_buffer(cmd)?;
        }
        self.submit_and_wait(cmd, &mut dispatch.frame)?;

        Ok(Box::new(GpuTensorBuffer::from_allocated(
            Arc::clone(&self.context),
            self.device.clone(),
            shape,
            dtype,
            dst,
        )))
    }

    fn dispatch_sampled(
        &self,
        sampled: &VulkanSampledImage,
        options: &ProcessingOptions,
        shape: TensorShape,
        dtype: DataType,
    ) -> Result<Box<dyn TensorBuffer>, GpuError> {
        let dst_size = shape.byte_size(dtype) as u64;
        let ctx = self.context.as_ref();

        let mut dispatch = self.record_lock.lock();

        let (set_layout, pipeline_layout, pipeline) = if sampled.is_ycbcr() {
            let sampler = sampled.sampler();
            let key = sampler.as_raw();
            if let Some(cached) = dispatch.ycbcr_pipelines.get(&key) {
                (cached.set_layout, cached.layout, cached.pipeline)
            } else {
                let immutable = [sampler];
                let bindings = [
                    storage_binding(0, vk::DescriptorType::UNIFORM_BUFFER),
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(1)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                        .immutable_samplers(&immutable),
                    storage_binding(2, vk::DescriptorType::STORAGE_BUFFER),
                ];
                let vkd = ctx.device();
                let set_layout = unsafe {
                    vkd.create_descriptor_set_layout(
                        &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                        None,
                    )?
                };
                let pipeline_layout = unsafe {
                    vkd.create_pipeline_layout(
                        &vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout]),
                        None,
                    )?
                };
                let stage = vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(self.shader_module)
                    .name(c"convert_image");
                let pipeline = unsafe {
                    vkd.create_compute_pipelines(
                        vk::PipelineCache::null(),
                        &[vk::ComputePipelineCreateInfo::default()
                            .stage(stage)
                            .layout(pipeline_layout)],
                        None,
                    )
                        .map_err(|(_, err)| err)?[0]
                };
                dispatch.ycbcr_pipelines.insert(
                    key,
                    YcbcrPipeline {
                        set_layout,
                        layout: pipeline_layout,
                        pipeline,
                    },
                );
                (set_layout, pipeline_layout, pipeline)
            }
        } else {
            (
                self.set_layout_image,
                self.layout_image,
                self.pipeline_image,
            )
        };

        let ubo_bytes = options_bytes(options);
        pool_buffer(
            ctx,
            &mut dispatch.frame.ubo,
            ubo_bytes.len() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
            "convert-ubo",
        )?;
        VulkanContext::write_allocation(
            &mut dispatch.frame.ubo.as_mut().unwrap().allocation,
            &ubo_bytes,
        )?;
        let ubo_buf = dispatch.frame.ubo.as_ref().unwrap().buffer;

        let dst = ctx.create_buffer(
            dst_size,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuOnly,
            "dst-ssbo",
        )?;

        let vkd = ctx.device();
        let desc_pool = *self.desc_pool.lock();
        // SAFETY: as in `dispatch_ssbo`; `record_lock` is held for this dispatch.
        unsafe { vkd.reset_descriptor_pool(desc_pool, vk::DescriptorPoolResetFlags::empty())? };
        let set = unsafe {
            vkd.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(desc_pool)
                    .set_layouts(&[set_layout]),
            )?
        }[0];

        let ubo_info = vk::DescriptorBufferInfo::default()
            .buffer(ubo_buf)
            .offset(0)
            .range(ubo_bytes.len() as u64);
        let image_info = vk::DescriptorImageInfo::default()
            .sampler(if sampled.is_ycbcr() {
                vk::Sampler::null()
            } else {
                self.sampler
            })
            .image_view(sampled.view())
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let dst_info = vk::DescriptorBufferInfo::default()
            .buffer(dst.buffer)
            .offset(0)
            .range(dst.size);
        let writes = [
            buffer_write(set, 0, vk::DescriptorType::UNIFORM_BUFFER, &ubo_info),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&image_info)),
            buffer_write(set, 2, vk::DescriptorType::STORAGE_BUFFER, &dst_info),
        ];
        unsafe { vkd.update_descriptor_sets(&writes, &[]) };

        let cmd = self.alloc_cmd()?;
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { vkd.begin_command_buffer(cmd, &begin)? };
        let src_queue = if sampled.acquire_from_external() {
            vk::QUEUE_FAMILY_EXTERNAL
        } else {
            vk::QUEUE_FAMILY_IGNORED
        };
        let barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(src_queue)
            .dst_queue_family_index(if sampled.acquire_from_external() {
                ctx.queue_family_index()
            } else {
                vk::QUEUE_FAMILY_IGNORED
            })
            .image(sampled.image())
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
            vkd.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            vkd.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &[set],
                &[],
            );
            vkd.cmd_dispatch(cmd, options.dest_w.div_ceil(16), options.dest_h.div_ceil(16), 1);
            buffer_barrier(
                vkd,
                cmd,
                dst.buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::SHADER_WRITE,
                vk::AccessFlags::TRANSFER_READ,
            );
            vkd.end_command_buffer(cmd)?;
        }
        self.submit_and_wait(cmd, &mut dispatch.frame)?;

        Ok(Box::new(GpuTensorBuffer::from_allocated(
            Arc::clone(&self.context),
            self.device.clone(),
            shape,
            dtype,
            dst,
        )))
    }

    fn alloc_cmd(&self) -> Result<vk::CommandBuffer, GpuError> {
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        Ok(unsafe { self.context.device().allocate_command_buffers(&info) }?[0])
    }

    fn submit_and_wait(
        &self,
        cmd: vk::CommandBuffer,
        frame: &mut FramePool,
    ) -> Result<(), GpuError> {
        let vkd = self.context.device();
        // SAFETY: `frame.fence` belongs to this processor's device and is only
        // used while `record_lock` is held.
        unsafe { vkd.reset_fences(&[frame.fence])? };

        let result = self
            .context
            .submit(cmd, frame.fence)
            .and_then(|()| self.context.wait_fence(frame.fence));

        // SAFETY: if submit failed nothing is pending; if it succeeded the wait
        // above has completed, so the command buffer is no longer in use.
        unsafe {
            vkd.free_command_buffers(self.cmd_pool, &[cmd]);
        }

        result?;
        Ok(())
    }
}

impl Drop for GpuImageProcessor {
    fn drop(&mut self) {
        let vkd = self.context.device();
        // SAFETY: waiting for idle ensures no submitted dispatch still uses the
        // pipelines, layouts or pools destroyed below.
        let _ = unsafe { vkd.device_wait_idle() };
        let mut dispatch = self.record_lock.lock();
        let ctx = self.context.as_ref();
        if let Some(buf) = dispatch.frame.staging_src.take() {
            ctx.destroy_buffer(buf);
        }
        if let Some(buf) = dispatch.frame.src.take() {
            ctx.destroy_buffer(buf);
        }
        if let Some(buf) = dispatch.frame.ubo.take() {
            ctx.destroy_buffer(buf);
        }
        // SAFETY: every handle below was created by `new` on this device and is
        // owned solely by this processor, so each is destroyed exactly once.
        unsafe {
            vkd.destroy_fence(dispatch.frame.fence, None);
            for cached in dispatch.ycbcr_pipelines.values() {
                vkd.destroy_pipeline(cached.pipeline, None);
                vkd.destroy_pipeline_layout(cached.layout, None);
                vkd.destroy_descriptor_set_layout(cached.set_layout, None);
            }
            vkd.destroy_pipeline(self.pipeline_ssbo, None);
            vkd.destroy_pipeline(self.pipeline_image, None);
            vkd.destroy_pipeline_layout(self.layout_ssbo, None);
            vkd.destroy_pipeline_layout(self.layout_image, None);
            vkd.destroy_descriptor_set_layout(self.set_layout_ssbo, None);
            vkd.destroy_descriptor_set_layout(self.set_layout_image, None);
            vkd.destroy_shader_module(self.shader_module, None);
            vkd.destroy_sampler(self.sampler, None);
            vkd.destroy_command_pool(self.cmd_pool, None);
            vkd.destroy_descriptor_pool(*self.desc_pool.lock(), None);
        }
    }
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

/// Number of 32-bit words in the shader's `ProcessingOptions` uniform block.
const OPTIONS_WORDS: usize = 13;

/// Vulkan guarantees uniform buffers may be bound at 64-byte granularity, so the
/// UBO is padded to that even though the payload is smaller.
const UBO_MIN_BYTES: usize = 64;

/// Serialise `options` into the uniform-buffer layout the shaders expect.
///
/// `processing_core::ProcessingOptions` is `#[repr(C)]` with twelve 32-bit
/// fields, matching the uniform block declared in `processing-shaders`. Writing
/// each field explicitly keeps that contract visible and avoids reinterpreting
/// the struct's bytes; the assertion below fails the build if a field is ever
/// added or resized without updating the shader side.
fn options_bytes(options: &ProcessingOptions) -> Vec<u8> {
    const _: () = assert!(
        std::mem::size_of::<ProcessingOptions>() == OPTIONS_WORDS * 4,
        "ProcessingOptions no longer matches the shader uniform block layout"
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

    let mut bytes = vec![0u8; (OPTIONS_WORDS * 4).max(UBO_MIN_BYTES)];
    for (slot, word) in bytes.chunks_exact_mut(4).zip(words) {
        slot.copy_from_slice(&word.to_le_bytes());
    }
    bytes
}
