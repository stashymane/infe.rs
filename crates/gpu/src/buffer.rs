use crate::context::VulkanContext;
use crate::error::GpuContextError;
use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use gpu_allocator::MemoryLocation;

/// Raw Vulkan buffer allocation metadata for external consumers (e.g. ExecuTorch staging copy).
#[derive(Clone, Copy, Debug)]
pub struct VulkanBufferHandle {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub offset: u64,
    pub size: u64,
}

/// Buffer plus memory allocated through the shared [`VulkanContext`] allocator.
pub struct AllocatedBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Allocation,
    pub size: u64,
}

impl AllocatedBuffer {
    pub fn vulkan_handle(&self) -> VulkanBufferHandle {
        VulkanBufferHandle {
            buffer: self.buffer,
            memory: unsafe { self.allocation.memory() },
            offset: self.allocation.offset(),
            size: self.size,
        }
    }
}

impl VulkanContext {
    pub fn create_buffer(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
        name: &str,
    ) -> Result<AllocatedBuffer, GpuContextError> {
        let size = size.max(4);
        let create_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { self.device().create_buffer(&create_info, None) }?;
        let requirements = unsafe { self.device().get_buffer_memory_requirements(buffer) };

        let allocation = self
            .allocator()
            .lock()
            .unwrap()
            .as_mut()
            .ok_or_else(|| GpuContextError::Other("allocator destroyed".into()))?
            .allocate(&AllocationCreateDesc {
                name,
                requirements,
                location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })?;

        unsafe {
            self.device().bind_buffer_memory(
                buffer,
                allocation.memory(),
                allocation.offset(),
            )?;
        }

        Ok(AllocatedBuffer {
            buffer,
            allocation,
            size,
        })
    }

    pub fn destroy_buffer(&self, buffer: AllocatedBuffer) {
        unsafe {
            self.device().destroy_buffer(buffer.buffer, None);
        }
        if let Some(allocator) = self.allocator().lock().unwrap().as_mut() {
            if let Err(err) = allocator.free(buffer.allocation) {
                eprintln!("infers-gpu: failed to free buffer allocation: {err}");
            }
        }
    }

    pub fn write_allocation(
        allocation: &mut Allocation,
        bytes: &[u8],
    ) -> Result<(), GpuContextError> {
        let mapped = allocation.mapped_slice_mut().ok_or_else(|| {
            GpuContextError::Other("allocation is not host-visible".into())
        })?;
        if mapped.len() < bytes.len() {
            return Err(GpuContextError::Other(format!(
                "mapped allocation too small: {} < {}",
                mapped.len(),
                bytes.len()
            )));
        }
        mapped[..bytes.len()].copy_from_slice(bytes);
        Ok(())
    }

    pub fn read_allocation(allocation: &Allocation, out: &mut [u8]) -> Result<(), GpuContextError> {
        let mapped = allocation.mapped_slice().ok_or_else(|| {
            GpuContextError::Other("allocation is not host-visible".into())
        })?;
        if mapped.len() < out.len() {
            return Err(GpuContextError::Other(format!(
                "mapped allocation too small: {} < {}",
                mapped.len(),
                out.len()
            )));
        }
        out.copy_from_slice(&mapped[..out.len()]);
        Ok(())
    }

    /// Copy between buffer regions on the shared compute queue.
    pub fn copy_buffer(
        &self,
        src: vk::Buffer,
        src_offset: u64,
        dst: vk::Buffer,
        dst_offset: u64,
        size: u64,
    ) -> Result<(), GpuContextError> {
        let device = self.device();
        let cmd_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(self.queue_family_index())
                    .flags(
                        vk::CommandPoolCreateFlags::TRANSIENT
                            | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                    ),
                None,
            )?
        };

        let cmd = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?
        }[0];

        unsafe {
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            buffer_barrier(
                device,
                cmd,
                src,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                vk::AccessFlags::TRANSFER_READ,
            );
            device.cmd_copy_buffer(
                cmd,
                src,
                dst,
                &[vk::BufferCopy::default()
                    .src_offset(src_offset)
                    .dst_offset(dst_offset)
                    .size(size)],
            );
            buffer_barrier(
                device,
                cmd,
                dst,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ,
            );
            device.end_command_buffer(cmd)?;
        }

        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }?;
        self.submit(cmd, fence)?;
        self.wait_fence(fence)?;
        unsafe {
            device.destroy_fence(fence, None);
            device.destroy_command_pool(cmd_pool, None);
        }
        Ok(())
    }
}

unsafe fn buffer_barrier(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    buffer: vk::Buffer,
    src_stage: vk::PipelineStageFlags,
    dst_stage: vk::PipelineStageFlags,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
) {
    let barrier = vk::BufferMemoryBarrier::default()
        .buffer(buffer)
        .size(vk::WHOLE_SIZE)
        .src_access_mask(src_access)
        .dst_access_mask(dst_access);
    unsafe {
        device.cmd_pipeline_barrier(
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
