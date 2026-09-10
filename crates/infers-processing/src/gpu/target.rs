use super::resources::OutputDestination;
use super::staging::SessionStagingQuery;
use super::util::pool_buffer;
use infers_core::{CoreError, DataType, MaterializeTarget, Tensor, TensorShape};
use infers_gpu::GpuError;
use infers_gpu::{
    ash::vk, gpu_allocator::MemoryLocation, tensor_from_allocated, tensor_from_external,
    AllocatedBuffer, Vulkan, VulkanContext,
};

pub(crate) fn resolve_materialize_target(
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

pub(crate) fn finish_materialized_tensor(
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
        OutputDestination::External(handle) => {
            Ok(tensor_from_external(vulkan, shape, dtype, handle))
        }
    }
}
