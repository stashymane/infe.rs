use crate::error::GpuError;
use infers_core::{AnyHostTensor, CpuTensor, DataType, Device, TensorBuffer, TensorShape};
use infers_gpu::ash::vk;
use infers_gpu::gpu_allocator::MemoryLocation;
use infers_gpu::{AllocatedBuffer, VulkanContext};
use std::sync::Arc;

struct GpuTensorInner {
    context: Arc<VulkanContext>,
    buffer: Option<AllocatedBuffer>,
}

impl Drop for GpuTensorInner {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.context.destroy_buffer(buffer);
        }
    }
}

/// GPU-resident tensor stored in a `VkBuffer` on the shared Vulkan context.
#[derive(Clone)]
pub struct GpuTensorBuffer {
    device: Device,
    shape: TensorShape,
    dtype: DataType,
    inner: Arc<GpuTensorInner>,
}

impl std::fmt::Debug for GpuTensorBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuTensorBuffer")
            .field("device", &self.device)
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("byte_size", &self.inner.buffer.as_ref().map(|b| b.size))
            .finish()
    }
}

impl GpuTensorBuffer {
    pub(crate) fn from_allocated(
        context: Arc<VulkanContext>,
        device: Device,
        shape: TensorShape,
        dtype: DataType,
        buffer: AllocatedBuffer,
    ) -> Self {
        Self {
            device,
            shape,
            dtype,
            inner: Arc::new(GpuTensorInner {
                context,
                buffer: Some(buffer),
            }),
        }
    }

    /// Shared Vulkan context that owns this tensor's buffer.
    pub fn context(&self) -> &Arc<VulkanContext> {
        &self.inner.context
    }

    /// Underlying `VkBuffer` for future zero-copy handoff into ExecuTorch Vulkan.
    pub fn vk_buffer(&self) -> Option<vk::Buffer> {
        self.inner.buffer.as_ref().map(|b| b.buffer)
    }
}

impl TensorBuffer for GpuTensorBuffer {
    fn shape(&self) -> &TensorShape {
        &self.shape
    }

    fn dtype(&self) -> DataType {
        self.dtype
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn read_to_cpu(&self) -> Result<Box<dyn AnyHostTensor>, infers_core::CoreError> {
        let src = self.inner.buffer.as_ref().ok_or_else(|| {
            infers_core::CoreError::BufferTransferFailed("GPU buffer already destroyed".into())
        })?;
        let bytes = read_buffer_to_vec(&self.inner.context, src).map_err(|err| {
            infers_core::CoreError::BufferTransferFailed(err.to_string())
        })?;
        host_tensor(self.dtype, self.shape.clone(), bytes)
    }

    fn copy_to_device(&self, target: &Device) -> Result<Box<dyn TensorBuffer>, infers_core::CoreError> {
        if target == &self.device {
            Ok(Box::new(self.clone()))
        } else if target.is_cpu() {
            let host_tensor = self.read_to_cpu()?;
            match self.dtype {
                DataType::U8 => {
                    let tensor = host_tensor
                        .as_any()
                        .downcast_ref::<CpuTensor<u8>>()
                        .cloned()
                        .ok_or_else(|| {
                            infers_core::CoreError::BufferTransferFailed(
                                "Failed to downcast u8 tensor".into(),
                            )
                        })?;
                    Ok(Box::new(tensor))
                }
                DataType::F32 => {
                    let tensor = host_tensor
                        .as_any()
                        .downcast_ref::<CpuTensor<f32>>()
                        .cloned()
                        .ok_or_else(|| {
                            infers_core::CoreError::BufferTransferFailed(
                                "Failed to downcast f32 tensor".into(),
                            )
                        })?;
                    Ok(Box::new(tensor))
                }
                _ => Err(infers_core::CoreError::BufferTransferFailed(format!(
                    "Unsupported copy for dtype {:?}",
                    self.dtype
                ))),
            }
        } else {
            Err(infers_core::CoreError::BufferTransferFailed(format!(
                "Cross-device GPU copy from {} to {} is not supported",
                self.device, target
            )))
        }
    }
}

fn read_buffer_to_vec(
    context: &VulkanContext,
    src: &AllocatedBuffer,
) -> Result<Vec<u8>, GpuError> {
    let staging = context.create_buffer(
        src.size,
        vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::GpuToCpu,
        "readback-staging",
    )?;

    let device = context.device();
    let cmd_pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(context.queue_family_index())
        .flags(vk::CommandPoolCreateFlags::TRANSIENT);
    let cmd_pool = unsafe { device.create_command_pool(&cmd_pool_info, None) }?;
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(cmd_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cmd = unsafe { device.allocate_command_buffers(&alloc_info) }?[0];
    let begin = vk::CommandBufferBeginInfo::default()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe { device.begin_command_buffer(cmd, &begin) }?;
    let region = vk::BufferCopy::default().size(src.size);
    unsafe {
        device.cmd_copy_buffer(cmd, src.buffer, staging.buffer, &[region]);
        device.end_command_buffer(cmd)?;
    }
    let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }?;
    context.submit(cmd, fence)?;
    context.wait_fence(fence)?;
    unsafe {
        device.destroy_fence(fence, None);
        device.destroy_command_pool(cmd_pool, None);
    }

    let mut out = vec![0u8; src.size as usize];
    VulkanContext::read_allocation(&staging.allocation, &mut out)?;
    context.destroy_buffer(staging);
    Ok(out)
}

fn host_tensor(
    dtype: DataType,
    shape: TensorShape,
    bytes: Vec<u8>,
) -> Result<Box<dyn AnyHostTensor>, infers_core::CoreError> {
    match dtype {
        DataType::U8 => {
            let expected = shape.byte_size(dtype);
            let data = bytes.into_iter().take(expected).collect();
            Ok(Box::new(CpuTensor::from_u8(shape, data)?))
        }
        DataType::F32 => {
            let count = shape.element_count();
            let mut f32_vec = Vec::with_capacity(count);
            for chunk in bytes.chunks_exact(4).take(count) {
                let bits = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                f32_vec.push(f32::from_bits(bits));
            }
            Ok(Box::new(CpuTensor::from_f32(shape, f32_vec)?))
        }
        other => Err(infers_core::CoreError::InvalidDataType {
            expected: DataType::F32,
            actual: other,
        }),
    }
}
