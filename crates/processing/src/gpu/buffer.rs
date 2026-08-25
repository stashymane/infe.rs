use infers_core::CoreError;
use infers_gpu::GpuError;
use infers_core::{AnyHostTensor, CpuTensor, DataType, Device, TensorBuffer, TensorShape};
use infers_gpu::ash::vk;
use infers_gpu::gpu_allocator::MemoryLocation;
use infers_gpu::{AllocatedBuffer, VulkanBufferHandle, VulkanContext};
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

    /// Buffer, device memory, offset, and byte size for GPU→GPU copies into delegate staging.
    pub fn vulkan_handle(&self) -> Option<VulkanBufferHandle> {
        self.inner.buffer.as_ref().map(|b| b.vulkan_handle())
    }

    /// Allocate a zero-initialized buffer on `context` for `shape` / `dtype`.
    pub fn allocate(
        context: Arc<VulkanContext>,
        shape: TensorShape,
        dtype: DataType,
    ) -> Result<Self, CoreError> {
        Self::try_allocate(context, shape, dtype).map_err(CoreError::from)
    }

    fn try_allocate(
        context: Arc<VulkanContext>,
        shape: TensorShape,
        dtype: DataType,
    ) -> Result<Self, GpuError> {
        use infers_gpu::gpu_allocator::MemoryLocation;
        let bytes = shape.byte_size(dtype) as u64;
        let buffer = context
            .create_buffer(
                bytes,
                vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::CpuToGpu,
                "gpu-tensor",
            )?;
        Ok(Self::from_allocated(
            Arc::clone(&context),
            context.logical_device().clone(),
            shape,
            dtype,
            buffer,
        ))
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

    fn as_any(&self) -> &dyn std::any::Any {
        self
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
            match self.dtype {
                DataType::U8 => {
                    let host = self.read_to_cpu()?;
                    let slice = host.as_slice_u8()?;
                    let tensor = CpuTensor::from_u8(self.shape.clone(), slice.to_vec())?;
                    Ok(Box::new(tensor))
                }
                DataType::F32 => {
                    let host = self.read_to_cpu()?;
                    let slice = host.as_slice_f32()?;
                    let tensor = CpuTensor::from_f32(self.shape.clone(), slice.to_vec())?;
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

    context
        .copy_buffer(src.buffer, 0, staging.buffer, 0, src.size)?;

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
