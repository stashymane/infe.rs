use crate::context::VulkanContext;
use crate::error::GpuContextError;
use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use gpu_allocator::MemoryLocation;

/// Buffer plus memory allocated through the shared [`VulkanContext`] allocator.
pub struct AllocatedBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Allocation,
    pub size: u64,
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
        data: &[u8],
    ) -> Result<(), GpuContextError> {
        let mapped = allocation.mapped_slice_mut().ok_or_else(|| {
            GpuContextError::Other("allocation is not host-visible".into())
        })?;
        if mapped.len() < data.len() {
            return Err(GpuContextError::Other(format!(
                "mapped allocation too small: {} < {}",
                mapped.len(),
                data.len()
            )));
        }
        mapped[..data.len()].copy_from_slice(data);
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
}
