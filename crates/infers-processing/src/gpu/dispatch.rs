use super::resources::{
    DispatchResources, GpuFrameSlot, GpuImageProcessorInner, GpuProcessInput, SampledDispatchInfo,
    SampledPipelineBinding, YcbcrPipeline,
};
use super::staging::{staging_for_materialize, SessionStagingQuery};
use super::target::{finish_materialized_tensor, resolve_materialize_target};
use super::util::{
    buffer_write, output_shape_dtype, reset_and_record, storage_binding, write_options_bytes,
    PUSH_CONSTANT_BYTES,
};
use infers_core::{
    CoreError, DataType, ImageFormat, MaterializeTarget, ProcessingOptions, Tensor, TensorShape,
};
use infers_gpu::GpuError;
use infers_gpu::{ash::vk, buffer_barrier, SharedYcbcrSampler, Vulkan, VulkanSampledImage};
use std::sync::Arc;

impl GpuImageProcessorInner {
    pub(crate) fn take_upload_buffer(
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

    pub(crate) fn store_upload_buffer(
        &self,
        width: u32,
        height: u32,
        format: ImageFormat,
        buffer: infers_gpu::VulkanImageBuffer,
    ) {
        *self.upload_buffer.lock() = Some((width, height, format, buffer));
    }

    pub(crate) fn materialize(
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
        let slot = &mut dispatch.slot;

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
        let slot = &mut dispatch.slot;

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
        let dispatch_info = SampledDispatchInfo::from(sampled);
        let binding = self.resolve_sampled_binding(&mut dispatch, sampled)?;
        let slot = &mut dispatch.slot;

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
                    ctx.external_acquire_queue_family()
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
        if dispatch_info.current_layout != vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL {
            sampled.set_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        }
        finish_materialized_tensor(
            &self.vulkan,
            shape,
            dtype,
            destination,
            slot.dst.take(),
        )
    }

    fn resolve_sampled_binding(
        &self,
        dispatch: &mut DispatchResources,
        sampled: &Arc<VulkanSampledImage>,
    ) -> Result<SampledPipelineBinding, GpuError> {
        if !sampled.is_ycbcr() {
            return Ok(SampledPipelineBinding {
                pipeline_layout: self.layout_image,
                pipeline: self.pipeline_image,
                desc_set: dispatch.slot.desc_image,
            });
        }

        let ycbcr = sampled.shared_ycbcr().ok_or_else(|| {
            GpuError::Other("YCbCr sampled image missing shared sampler".into())
        })?;
        let key = ycbcr.sampler_key();

        if let Some(cached) = dispatch.ycbcr_pipelines.get(&key) {
            return Ok(SampledPipelineBinding {
                pipeline_layout: cached.layout,
                pipeline: cached.pipeline,
                desc_set: cached.desc_set,
            });
        }

        self.create_ycbcr_pipeline(dispatch, ycbcr, key)
    }

    fn create_ycbcr_pipeline(
        &self,
        dispatch: &mut DispatchResources,
        ycbcr: &Arc<SharedYcbcrSampler>,
        key: u64,
    ) -> Result<SampledPipelineBinding, GpuError> {
        let vkd = self.context.device();
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
        let desc_set = unsafe {
            vkd.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.desc_pool)
                    .set_layouts(&[set_layout]),
            )?
        }[0];
        dispatch.ycbcr_pipelines.insert(
            key,
            YcbcrPipeline {
                _ycbcr: Arc::clone(ycbcr),
                set_layout,
                layout: pipeline_layout,
                pipeline,
                desc_set,
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
