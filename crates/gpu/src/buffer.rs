use crate::context::VulkanContext;
use crate::error::GpuError;
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
            // SAFETY: `gpu_allocator` marks this accessor unsafe because the
            // caller must not free or alias the memory behind the allocator's
            // back. The handle is metadata only; ownership stays with `self`.
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
    ) -> Result<AllocatedBuffer, GpuError> {
        let size = size.max(4);
        let create_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        // SAFETY: `create_info` is fully initialised and borrows nothing beyond
        // the call.
        let buffer = unsafe { self.device().create_buffer(&create_info, None) }?;
        // SAFETY: `buffer` was just created on this device.
        let requirements = unsafe { self.device().get_buffer_memory_requirements(buffer) };

        let allocation = match self
            .allocator()
            .lock()
            .as_mut()
            .ok_or_else(|| GpuError::Other("allocator destroyed".into()))
            .and_then(|allocator| {
                allocator
                    .allocate(&AllocationCreateDesc {
                        name,
                        requirements,
                        location,
                        linear: true,
                        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                    })
                    .map_err(GpuError::from)
            }) {
            Ok(allocation) => allocation,
            Err(err) => {
                // SAFETY: `buffer` has no memory bound and no other owner.
                unsafe { self.device().destroy_buffer(buffer, None) };
                return Err(err);
            }
        };

        // SAFETY: `buffer` has no memory bound yet, and the allocation came from
        // this device's allocator sized to `buffer`'s requirements.
        unsafe {
            self.device()
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
        }

        Ok(AllocatedBuffer {
            buffer,
            allocation,
            size,
        })
    }

    pub fn destroy_buffer(&self, buffer: AllocatedBuffer) {
        // SAFETY: `buffer` is owned by value, so nothing else references it. The
        // caller is responsible for ensuring no submitted work still uses it.
        unsafe {
            self.device().destroy_buffer(buffer.buffer, None);
        }
        if let Some(allocator) = self.allocator().lock().as_mut()
            && let Err(err) = allocator.free(buffer.allocation)
        {
            eprintln!("infers-gpu: failed to free buffer allocation: {err}");
        }
    }

    pub fn write_allocation(
        allocation: &mut Allocation,
        bytes: &[u8],
    ) -> Result<(), GpuError> {
        let mapped = allocation.mapped_slice_mut().ok_or_else(|| {
            GpuError::Other("allocation is not host-visible".into())
        })?;
        if mapped.len() < bytes.len() {
            return Err(GpuError::Other(format!(
                "mapped allocation too small: {} < {}",
                mapped.len(),
                bytes.len()
            )));
        }
        mapped[..bytes.len()].copy_from_slice(bytes);
        Ok(())
    }

    pub fn read_allocation(allocation: &Allocation, out: &mut [u8]) -> Result<(), GpuError> {
        let mapped = allocation.mapped_slice().ok_or_else(|| {
            GpuError::Other("allocation is not host-visible".into())
        })?;
        if mapped.len() < out.len() {
            return Err(GpuError::Other(format!(
                "mapped allocation too small: {} < {}",
                mapped.len(),
                out.len()
            )));
        }
        out.copy_from_slice(&mapped[..out.len()]);
        Ok(())
    }

    /// Copy between buffer regions on the shared compute queue, blocking until
    /// the copy has completed.
    pub fn copy_buffer(
        &self,
        src: vk::Buffer,
        src_offset: u64,
        dst: vk::Buffer,
        dst_offset: u64,
        size: u64,
    ) -> Result<(), GpuError> {
        self.record_and_wait(|device, cmd| {
            // SAFETY: `cmd` is recording, and `src`/`dst` belong to this device.
            // The caller guarantees both regions cover `size` bytes at the given
            // offsets. The barriers make a prior compute write to `src` visible
            // to the transfer, and the transfer write to `dst` visible to later
            // compute and transfer reads.
            unsafe {
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
            }
            Ok(())
        })
    }
}

/// Insert a full-buffer memory dependency into `cmd`.
///
/// # Safety
///
/// `cmd` must be in the recording state and `buffer` must belong to `device`.
pub unsafe fn buffer_barrier(
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
    // SAFETY: guaranteed by this function's contract; `barrier` outlives the call
    // and covers the whole buffer, which is always a valid range.
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
