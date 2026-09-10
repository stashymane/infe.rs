use infers_core::{CoreError, ImageFormat};
use infers_gpu::GpuError;
use infers_gpu::{
    ash, ash::vk, AllocatedBuffer, SharedYcbcrSampler, Vulkan, VulkanBufferHandle, VulkanContext,
    VulkanImage, VulkanSampledImage,
};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone)]
pub(crate) enum GpuProcessInput {
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
pub(crate) struct SampledDispatchInfo {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub is_ycbcr: bool,
    pub acquire_from_external: bool,
    pub current_layout: vk::ImageLayout,
}

impl GpuProcessInput {
    pub(crate) fn from_image(input: &VulkanImage) -> Result<Self, CoreError> {
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

    pub(crate) fn linear_unstaged(
        buf: &infers_gpu::VulkanImageBuffer,
    ) -> Result<Self, CoreError> {
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

pub(crate) enum OutputDestination {
    Pooled,
    External(VulkanBufferHandle),
}

pub(crate) struct YcbcrPipeline {
    /// Keeps the immutable sampler alive for `set_layout`.
    pub _ycbcr: Arc<SharedYcbcrSampler>,
    pub set_layout: vk::DescriptorSetLayout,
    pub layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub desc_set: vk::DescriptorSet,
}

pub(crate) struct SampledPipelineBinding {
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub desc_set: vk::DescriptorSet,
}

pub(crate) struct GpuFrameSlot {
    pub dst: Option<AllocatedBuffer>,
    pub cmd: vk::CommandBuffer,
    pub desc_ssbo: vk::DescriptorSet,
    pub desc_image: vk::DescriptorSet,
    pub fence: vk::Fence,
}

pub(crate) struct DispatchResources {
    pub slot: GpuFrameSlot,
    pub ycbcr_pipelines: HashMap<u64, YcbcrPipeline>,
}

pub(crate) struct GpuImageProcessorInner {
    pub vulkan: Vulkan,
    pub context: Arc<VulkanContext>,
    pub shader_module: vk::ShaderModule,
    pub pipeline_cache: vk::PipelineCache,
    pub set_layout_ssbo: vk::DescriptorSetLayout,
    pub set_layout_image: vk::DescriptorSetLayout,
    pub layout_ssbo: vk::PipelineLayout,
    pub layout_image: vk::PipelineLayout,
    pub pipeline_ssbo: vk::Pipeline,
    pub pipeline_image: vk::Pipeline,
    pub sampler: vk::Sampler,
    pub cmd_pool: vk::CommandPool,
    pub desc_pool: vk::DescriptorPool,
    pub record_lock: Mutex<DispatchResources>,
    /// Reused SSBO upload buffer for hardware-sourced frames.
    pub upload_buffer: Mutex<Option<(u32, u32, ImageFormat, infers_gpu::VulkanImageBuffer)>>,
}

pub(crate) fn create_frame_slot(
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

pub(crate) fn drop_gpu_processor_inner(inner: &mut GpuImageProcessorInner) {
    let vkd = inner.context.device();
    let _ = unsafe { vkd.device_wait_idle() };
    let mut dispatch = inner.record_lock.lock();
    let ctx = inner.context.as_ref();
    if let Some(buf) = dispatch.slot.dst.take() {
        ctx.destroy_buffer(buf);
    }
    unsafe {
        vkd.destroy_fence(dispatch.slot.fence, None);
        vkd.free_command_buffers(inner.cmd_pool, &[dispatch.slot.cmd]);
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
