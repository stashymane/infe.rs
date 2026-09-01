//! GPU NHWC→NCHW layout transpose compute pass.

use super::SHADERS;
use infers_core::{CoreError, DataType, Tensor, TensorShape};
use infers_gpu::GpuError;
use infers_gpu::ash::vk;
use infers_gpu::gpu_allocator::MemoryLocation;
use infers_gpu::{buffer_barrier, tensor_from_allocated, Vulkan, VulkanContext};
use parking_lot::Mutex;
use std::sync::Arc;

#[repr(C)]
struct LayoutDims {
    h: u32,
    w: u32,
}

struct LayoutPipeline {
    shader_module: vk::ShaderModule,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    cmd_pool: vk::CommandPool,
    desc_pool: vk::DescriptorPool,
    fence: vk::Fence,
}

/// Cached GPU layout transpose pipelines keyed by Vulkan context identity.
pub struct LayoutGpuPass {
    context: Arc<VulkanContext>,
    pipeline: Mutex<Option<LayoutPipeline>>,
}

impl LayoutGpuPass {
    pub fn new(context: Arc<VulkanContext>) -> Self {
        Self {
            context,
            pipeline: Mutex::new(None),
        }
    }

    pub fn nhwc_to_nchw(&self, input: &Tensor<Vulkan>) -> Result<Tensor<Vulkan>, CoreError> {
        if !Arc::ptr_eq(&self.context, input.device().context()) {
            return Err(CoreError::DeviceMismatch {
                expected: self.context.device_info().clone(),
                actual: input.device().info().clone(),
            });
        }
        self.try_nhwc_to_nchw(input).map_err(CoreError::from)
    }

    fn try_nhwc_to_nchw(&self, input: &Tensor<Vulkan>) -> Result<Tensor<Vulkan>, GpuError> {
        let dims = input.shape().dims();
        if dims.len() != 4 || dims[0] != 1 || dims[3] != 3 {
            return Err(GpuError::Other(format!(
                "expected NHWC [1, H, W, 3], got {dims:?}"
            )));
        }
        if input.dtype() != DataType::F32 {
            return Err(GpuError::Other(format!(
                "layout transpose expects F32, got {:?}",
                input.dtype()
            )));
        }
        let h = dims[1];
        let w = dims[2];
        let out_shape = TensorShape::new([1, 3, h, w])
            .map_err(|err| GpuError::Other(err.to_string()))?;
        let src = input.storage().vulkan_handle().ok_or_else(|| {
            GpuError::Other("GPU input buffer already destroyed".into())
        })?;

        let mut guard = self.pipeline.lock();
        if guard.is_none() {
            *guard = Some(create_layout_pipeline(&self.context)?);
        }
        let pipe = guard.as_mut().expect("pipeline initialized");

        let ctx = self.context.as_ref();
        let dst = ctx.create_buffer(
            out_shape.byte_size(DataType::F32) as u64,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuOnly,
            "layout-nchw-dst",
        )?;

        let uniform = LayoutDims {
            h: h as u32,
            w: w as u32,
        };
        let mut ubo = ctx.create_buffer(
            64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
            "layout-dims-ubo",
        )?;
        let ubo_bytes = layout_dims_bytes(&uniform);
        infers_gpu::VulkanContext::write_allocation(&mut ubo.allocation, &ubo_bytes)?;

        let vkd = ctx.device();
        unsafe { vkd.reset_fences(&[pipe.fence])? };
        unsafe {
            vkd.reset_descriptor_pool(pipe.desc_pool, vk::DescriptorPoolResetFlags::empty())?
        };
        let set = unsafe {
            vkd.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pipe.desc_pool)
                    .set_layouts(&[pipe.set_layout]),
            )?
        }[0];

        let ubo_info = vk::DescriptorBufferInfo::default()
            .buffer(ubo.buffer)
            .range(64);
        let src_info = vk::DescriptorBufferInfo::default()
            .buffer(src.buffer)
            .offset(src.offset)
            .range(src.size);
        let dst_info = vk::DescriptorBufferInfo::default()
            .buffer(dst.buffer)
            .range(dst.size);
        let writes = [
            buffer_write(set, 0, vk::DescriptorType::UNIFORM_BUFFER, &ubo_info),
            buffer_write(set, 1, vk::DescriptorType::STORAGE_BUFFER, &src_info),
            buffer_write(set, 2, vk::DescriptorType::STORAGE_BUFFER, &dst_info),
        ];
        unsafe { vkd.update_descriptor_sets(&writes, &[]) };

        let cmd = unsafe {
            vkd.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pipe.cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?
        }[0];
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { vkd.begin_command_buffer(cmd, &begin)? };
        unsafe {
            buffer_barrier(
                vkd,
                cmd,
                src.buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::TRANSFER_READ,
                vk::AccessFlags::SHADER_READ,
            );
            vkd.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe.pipeline);
            vkd.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipe.pipeline_layout,
                0,
                &[set],
                &[],
            );
            vkd.cmd_dispatch(cmd, w.div_ceil(16) as u32, h.div_ceil(16) as u32, 1);
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
        ctx.submit(cmd, pipe.fence)?;
        ctx.wait_fence(pipe.fence)?;
        unsafe {
            vkd.free_command_buffers(pipe.cmd_pool, &[cmd]);
        }
        ctx.destroy_buffer(ubo);

        Ok(tensor_from_allocated(input.device(), out_shape, DataType::F32, dst))
    }
}

pub fn nhwc_to_nchw_gpu(
    device: &Vulkan,
    input: &Tensor<Vulkan>,
) -> Result<Tensor<Vulkan>, CoreError> {
    if !Arc::ptr_eq(device.context(), input.device().context()) {
        return Err(CoreError::DeviceMismatch {
            expected: device.info().clone(),
            actual: input.device().info().clone(),
        });
    }
    LayoutGpuPass::new(Arc::clone(device.context())).nhwc_to_nchw(input)
}

fn create_layout_pipeline(context: &VulkanContext) -> Result<LayoutPipeline, GpuError> {
    let vkd = context.device();
    let words = spirv_words(SHADERS)?;
    let shader_module = unsafe {
        vkd.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)?
    };
    let set_layout = unsafe {
        vkd.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                storage_binding(0, vk::DescriptorType::UNIFORM_BUFFER),
                storage_binding(1, vk::DescriptorType::STORAGE_BUFFER),
                storage_binding(2, vk::DescriptorType::STORAGE_BUFFER),
            ]),
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
        .module(shader_module)
        .name(c"nhwc_to_nchw_main");
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
                .max_sets(4)
                .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                .pool_sizes(&[
                    vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::UNIFORM_BUFFER,
                        descriptor_count: 4,
                    },
                    vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::STORAGE_BUFFER,
                        descriptor_count: 8,
                    },
                ]),
            None,
        )?
    };
    let fence = unsafe { vkd.create_fence(&vk::FenceCreateInfo::default(), None)? };
    Ok(LayoutPipeline {
        shader_module,
        set_layout,
        pipeline_layout,
        pipeline,
        cmd_pool,
        desc_pool,
        fence,
    })
}

fn layout_dims_bytes(dims: &LayoutDims) -> Vec<u8> {
    let mut bytes = vec![0u8; 64];
    bytes[0..4].copy_from_slice(&dims.h.to_le_bytes());
    bytes[4..8].copy_from_slice(&dims.w.to_le_bytes());
    bytes
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

impl Drop for LayoutGpuPass {
    fn drop(&mut self) {
        let Some(pipe) = self.pipeline.get_mut().take() else {
            return;
        };
        let vkd = self.context.device();
        // SAFETY: wait for idle so no in-flight dispatch still uses handles below.
        let _ = unsafe { vkd.device_wait_idle() };
        // SAFETY: handles were created in `create_layout_pipeline` on this device.
        unsafe {
            vkd.destroy_pipeline(pipe.pipeline, None);
            vkd.destroy_pipeline_layout(pipe.pipeline_layout, None);
            vkd.destroy_descriptor_set_layout(pipe.set_layout, None);
            vkd.destroy_shader_module(pipe.shader_module, None);
            vkd.destroy_command_pool(pipe.cmd_pool, None);
            vkd.destroy_descriptor_pool(pipe.desc_pool, None);
            vkd.destroy_fence(pipe.fence, None);
        }
    }
}
