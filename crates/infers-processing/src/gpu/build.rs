use super::resources::{
    create_frame_slot, DispatchResources, GpuFrameSlot, GpuImageProcessorInner,
};
use super::util::{spirv_words, storage_binding, PUSH_CONSTANT_BYTES};
use super::SHADERS;
use infers_gpu::GpuError;
use infers_gpu::{ash, ash::vk, Vulkan};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

pub(crate) fn build_inner(vulkan: Vulkan) -> Result<GpuImageProcessorInner, GpuError> {
    let context = Arc::clone(vulkan.context());
    let vkd = context.device();

    let words = spirv_words(SHADERS)?;
    let mut guard = ProcessorBuildGuard {
        device: vkd,
        shader_module: vk::ShaderModule::null(),
        pipeline_cache: vk::PipelineCache::null(),
        set_layout_ssbo: vk::DescriptorSetLayout::null(),
        set_layout_image: vk::DescriptorSetLayout::null(),
        layout_ssbo: vk::PipelineLayout::null(),
        layout_image: vk::PipelineLayout::null(),
        pipeline_ssbo: vk::Pipeline::null(),
        pipeline_image: vk::Pipeline::null(),
        sampler: vk::Sampler::null(),
        cmd_pool: vk::CommandPool::null(),
        desc_pool: vk::DescriptorPool::null(),
        slot: None,
    };

    guard.shader_module = unsafe {
        vkd.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)?
    };

    guard.pipeline_cache = unsafe {
        vkd.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)?
    };

    guard.set_layout_ssbo = unsafe {
        vkd.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                storage_binding(0, vk::DescriptorType::STORAGE_BUFFER),
                storage_binding(1, vk::DescriptorType::STORAGE_BUFFER),
            ]),
            None,
        )?
    };
    guard.set_layout_image = unsafe {
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
    guard.layout_ssbo = unsafe {
        vkd.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&[guard.set_layout_ssbo])
                .push_constant_ranges(&[push_constant_range]),
            None,
        )?
    };
    guard.layout_image = unsafe {
        vkd.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&[guard.set_layout_image])
                .push_constant_ranges(&[push_constant_range]),
            None,
        )?
    };

    let stage_main = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(guard.shader_module)
        .name(c"convert_main");
    let stage_image = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(guard.shader_module)
        .name(c"convert_image");
    let infos = [
        vk::ComputePipelineCreateInfo::default()
            .stage(stage_main)
            .layout(guard.layout_ssbo),
        vk::ComputePipelineCreateInfo::default()
            .stage(stage_image)
            .layout(guard.layout_image),
    ];
    let pipelines = unsafe {
        vkd.create_compute_pipelines(guard.pipeline_cache, &infos, None)
            .map_err(|(_, err)| err)?
    };
    guard.pipeline_ssbo = pipelines[0];
    guard.pipeline_image = pipelines[1];

    guard.sampler = unsafe {
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

    guard.cmd_pool = unsafe {
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

    // One sync slot + headroom for a few YCbCr format variants.
    guard.desc_pool = unsafe {
        vkd.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(2 + 4 + 8)
                .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                .pool_sizes(&[
                    vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::STORAGE_BUFFER,
                        descriptor_count: 4 + 4 + 16,
                    },
                    vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        descriptor_count: 2 + 4 + 8,
                    },
                ]),
            None,
        )?
    };

    guard.slot = Some(create_frame_slot(
        vkd,
        guard.cmd_pool,
        guard.desc_pool,
        guard.set_layout_ssbo,
        guard.set_layout_image,
    )?);

    let built = guard.disarm();
    Ok(GpuImageProcessorInner {
        vulkan,
        context,
        shader_module: built.shader_module,
        pipeline_cache: built.pipeline_cache,
        set_layout_ssbo: built.set_layout_ssbo,
        set_layout_image: built.set_layout_image,
        layout_ssbo: built.layout_ssbo,
        layout_image: built.layout_image,
        pipeline_ssbo: built.pipeline_ssbo,
        pipeline_image: built.pipeline_image,
        sampler: built.sampler,
        cmd_pool: built.cmd_pool,
        desc_pool: built.desc_pool,
        record_lock: Mutex::new(DispatchResources {
            slot: built.slot,
            ycbcr_pipelines: HashMap::new(),
        }),
        upload_buffer: Mutex::new(None),
    })
}

/// Destroys partially-built processor Vulkan objects unless [`ProcessorBuildGuard::disarm`]
/// transfers ownership into [`GpuImageProcessorInner`].
struct ProcessorBuildGuard<'a> {
    device: &'a ash::Device,
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
    slot: Option<GpuFrameSlot>,
}

struct ProcessorBuilt {
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
    slot: GpuFrameSlot,
}

impl ProcessorBuildGuard<'_> {
    fn disarm(mut self) -> ProcessorBuilt {
        let built = ProcessorBuilt {
            shader_module: self.shader_module,
            pipeline_cache: self.pipeline_cache,
            set_layout_ssbo: self.set_layout_ssbo,
            set_layout_image: self.set_layout_image,
            layout_ssbo: self.layout_ssbo,
            layout_image: self.layout_image,
            pipeline_ssbo: self.pipeline_ssbo,
            pipeline_image: self.pipeline_image,
            sampler: self.sampler,
            cmd_pool: self.cmd_pool,
            desc_pool: self.desc_pool,
            slot: self.slot.take().expect("frame slot created before disarm"),
        };
        self.shader_module = vk::ShaderModule::null();
        self.pipeline_cache = vk::PipelineCache::null();
        self.set_layout_ssbo = vk::DescriptorSetLayout::null();
        self.set_layout_image = vk::DescriptorSetLayout::null();
        self.layout_ssbo = vk::PipelineLayout::null();
        self.layout_image = vk::PipelineLayout::null();
        self.pipeline_ssbo = vk::Pipeline::null();
        self.pipeline_image = vk::Pipeline::null();
        self.sampler = vk::Sampler::null();
        self.cmd_pool = vk::CommandPool::null();
        self.desc_pool = vk::DescriptorPool::null();
        built
    }
}

impl Drop for ProcessorBuildGuard<'_> {
    fn drop(&mut self) {
        let device = self.device;
        unsafe {
            if let Some(slot) = self.slot.take() {
                device.destroy_fence(slot.fence, None);
            }
            if self.pipeline_ssbo != vk::Pipeline::null() {
                device.destroy_pipeline(self.pipeline_ssbo, None);
            }
            if self.pipeline_image != vk::Pipeline::null() {
                device.destroy_pipeline(self.pipeline_image, None);
            }
            if self.layout_ssbo != vk::PipelineLayout::null() {
                device.destroy_pipeline_layout(self.layout_ssbo, None);
            }
            if self.layout_image != vk::PipelineLayout::null() {
                device.destroy_pipeline_layout(self.layout_image, None);
            }
            if self.set_layout_ssbo != vk::DescriptorSetLayout::null() {
                device.destroy_descriptor_set_layout(self.set_layout_ssbo, None);
            }
            if self.set_layout_image != vk::DescriptorSetLayout::null() {
                device.destroy_descriptor_set_layout(self.set_layout_image, None);
            }
            if self.pipeline_cache != vk::PipelineCache::null() {
                device.destroy_pipeline_cache(self.pipeline_cache, None);
            }
            if self.shader_module != vk::ShaderModule::null() {
                device.destroy_shader_module(self.shader_module, None);
            }
            if self.sampler != vk::Sampler::null() {
                device.destroy_sampler(self.sampler, None);
            }
            if self.cmd_pool != vk::CommandPool::null() {
                device.destroy_command_pool(self.cmd_pool, None);
            }
            if self.desc_pool != vk::DescriptorPool::null() {
                device.destroy_descriptor_pool(self.desc_pool, None);
            }
        }
    }
}
