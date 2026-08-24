use crate::SHADERS;
use crate::buffer::GpuTensorBuffer;
use crate::error::GpuError;
use infers_core::{
    DataType, Device, ImageFormat, ImageInputBuffer, ProcessingOptions, TensorBuffer, TensorShape,
};
use infers_gpu::ash::vk;
use infers_gpu::gpu_allocator::MemoryLocation;
use infers_gpu::{VulkanContext, VulkanSampledImage};
use std::ffi::CString;
use std::sync::{Arc, Mutex};

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
    record_lock: Mutex<()>,
}

impl GpuImageProcessor {
    pub fn new(context: Arc<VulkanContext>) -> Result<Self, GpuError> {
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

        let entry_main = CString::new("convert_main").unwrap();
        let entry_image = CString::new("convert_image").unwrap();
        let stage_main = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(&entry_main);
        let stage_image = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(&entry_image);
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
            record_lock: Mutex::new(()),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn context(&self) -> &Arc<VulkanContext> {
        &self.context
    }

    pub fn process(
        &self,
        input: &dyn ImageInputBuffer,
        options: &ProcessingOptions,
    ) -> Result<Box<dyn TensorBuffer>, GpuError> {
        let (dest_w, dest_h) = (options.dest_w, options.dest_h);
        if dest_w == 0 || dest_h == 0 {
            return Err(GpuError::InvalidBuffer(
                "Destination dimensions must be non-zero".into(),
            ));
        }
        if options.dest_format.is_yuv() {
            return Err(GpuError::InvalidBuffer(
                "GPU convert kernels emit RGB888 or RGBF32, not YUV".into(),
            ));
        }

        let src_format = input.format();
        if src_format.is_yuv() && (input.width() % 2 != 0 || input.height() % 2 != 0) {
            return Err(GpuError::InvalidBuffer(
                "YUV 4:2:0 input width and height must be even".into(),
            ));
        }

        let (_crop_x, _crop_y, crop_w, crop_h) = options.effective_crop();
        if crop_w == 0 || crop_h == 0 {
            return Err(GpuError::InvalidBuffer(
                "Effective crop width and height must be greater than zero".into(),
            ));
        }

        let shape = TensorShape::new(vec![1, dest_h as usize, dest_w as usize, 3])?;
        let dtype = match options.dest_format {
            ImageFormat::RGB888 => DataType::U8,
            ImageFormat::RGBF32 => DataType::F32,
            ImageFormat::NV12 | ImageFormat::I420 => {
                return Err(GpuError::InvalidBuffer(
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
                return Err(GpuError::InvalidBuffer(format!(
                    "YUV/RGB source size mismatch: expected {expected} bytes, got {}",
                    bytes.len()
                )));
            }
            return self.dispatch_ssbo(bytes, &kernel_opts, shape, dtype);
        }

        if let Some(sampled) = input.as_any().downcast_ref::<VulkanSampledImage>() {
            return self.dispatch_sampled(sampled, &kernel_opts, shape, dtype);
        }

        Err(GpuError::InvalidBuffer(
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

        let mut staging_src = ctx.create_buffer(
            src_bytes.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            "src-staging",
        )?;
        VulkanContext::write_allocation(&mut staging_src.allocation, src_bytes)?;

        let src = ctx.create_buffer(
            src_bytes.len() as u64,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuOnly,
            "src-ssbo",
        )?;
        let dst = ctx.create_buffer(
            dst_size,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuOnly,
            "dst-ssbo",
        )?;
        let mut ubo = ctx.create_buffer(
            256,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
            "convert-ubo",
        )?;
        let ubo_bytes = options_bytes(options);
        VulkanContext::write_allocation(&mut ubo.allocation, &ubo_bytes)?;

        let _guard = self.record_lock.lock().unwrap();
        let vkd = ctx.device();
        let desc_pool = *self.desc_pool.lock().unwrap();
        unsafe { vkd.reset_descriptor_pool(desc_pool, vk::DescriptorPoolResetFlags::empty())? };
        let set = unsafe {
            vkd.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(desc_pool)
                    .set_layouts(&[self.set_layout_ssbo]),
            )?
        }[0];

        let ubo_info = vk::DescriptorBufferInfo::default()
            .buffer(ubo.buffer)
            .offset(0)
            .range(ubo_bytes.len() as u64);
        let src_info = vk::DescriptorBufferInfo::default()
            .buffer(src.buffer)
            .offset(0)
            .range(src.size);
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
        let copy = vk::BufferCopy::default().size(src_bytes.len() as u64);
        unsafe {
            vkd.cmd_copy_buffer(cmd, staging_src.buffer, src.buffer, &[copy]);
            buffer_barrier(
                vkd,
                cmd,
                src.buffer,
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
        self.submit_and_wait(cmd)?;

        ctx.destroy_buffer(staging_src);
        ctx.destroy_buffer(src);
        ctx.destroy_buffer(ubo);

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
        let dst = ctx.create_buffer(
            dst_size,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuOnly,
            "dst-ssbo",
        )?;
        let mut ubo = ctx.create_buffer(
            256,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
            "convert-ubo",
        )?;
        let ubo_bytes = options_bytes(options);
        VulkanContext::write_allocation(&mut ubo.allocation, &ubo_bytes)?;

        let _guard = self.record_lock.lock().unwrap();
        let vkd = ctx.device();

        let (set_layout, pipeline_layout, pipeline, extra_destroy) = if sampled.is_ycbcr() {
            let immutable = [sampled.sampler()];
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
            let entry = CString::new("convert_image").unwrap();
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(self.shader_module)
                .name(&entry);
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
            (set_layout, pipeline_layout, pipeline, true)
        } else {
            (
                self.set_layout_image,
                self.layout_image,
                self.pipeline_image,
                false,
            )
        };

        let desc_pool = *self.desc_pool.lock().unwrap();
        unsafe { vkd.reset_descriptor_pool(desc_pool, vk::DescriptorPoolResetFlags::empty())? };
        let set = unsafe {
            vkd.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(desc_pool)
                    .set_layouts(&[set_layout]),
            )?
        }[0];

        let ubo_info = vk::DescriptorBufferInfo::default()
            .buffer(ubo.buffer)
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
        self.submit_and_wait(cmd)?;

        if extra_destroy {
            unsafe {
                vkd.destroy_pipeline(pipeline, None);
                vkd.destroy_pipeline_layout(pipeline_layout, None);
                vkd.destroy_descriptor_set_layout(set_layout, None);
            }
        }
        ctx.destroy_buffer(ubo);

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

    fn submit_and_wait(&self, cmd: vk::CommandBuffer) -> Result<(), GpuError> {
        let vkd = self.context.device();
        let fence = unsafe { vkd.create_fence(&vk::FenceCreateInfo::default(), None) }?;
        self.context.submit(cmd, fence)?;
        self.context.wait_fence(fence)?;
        unsafe {
            vkd.destroy_fence(fence, None);
            vkd.free_command_buffers(self.cmd_pool, &[cmd]);
        }
        Ok(())
    }
}

impl Drop for GpuImageProcessor {
    fn drop(&mut self) {
        let vkd = self.context.device();
        let _ = unsafe { vkd.device_wait_idle() };
        unsafe {
            vkd.destroy_pipeline(self.pipeline_ssbo, None);
            vkd.destroy_pipeline(self.pipeline_image, None);
            vkd.destroy_pipeline_layout(self.layout_ssbo, None);
            vkd.destroy_pipeline_layout(self.layout_image, None);
            vkd.destroy_descriptor_set_layout(self.set_layout_ssbo, None);
            vkd.destroy_descriptor_set_layout(self.set_layout_image, None);
            vkd.destroy_shader_module(self.shader_module, None);
            vkd.destroy_sampler(self.sampler, None);
            vkd.destroy_command_pool(self.cmd_pool, None);
            vkd.destroy_descriptor_pool(*self.desc_pool.lock().unwrap(), None);
        }
    }
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

fn buffer_barrier(
    vkd: &ash::Device,
    cmd: vk::CommandBuffer,
    buffer: vk::Buffer,
    src_stage: vk::PipelineStageFlags,
    dst_stage: vk::PipelineStageFlags,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
) {
    let barrier = vk::BufferMemoryBarrier::default()
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(buffer)
        .offset(0)
        .size(vk::WHOLE_SIZE);
    unsafe {
        vkd.cmd_pipeline_barrier(
            cmd,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[barrier],
            &[],
        );
    }
}

fn spirv_words(bytes: &[u8]) -> Result<Vec<u32>, GpuError> {
    if bytes.len() % 4 != 0 {
        return Err(GpuError::Gpu("SPIR-V binary is not 4-byte aligned".into()));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn options_bytes(options: &ProcessingOptions) -> Vec<u8> {
    let size = std::mem::size_of::<ProcessingOptions>();
    let mut bytes = vec![0u8; size.max(64)];
    unsafe {
        std::ptr::copy_nonoverlapping(
            options as *const ProcessingOptions as *const u8,
            bytes.as_mut_ptr(),
            size,
        );
    }
    bytes
}
