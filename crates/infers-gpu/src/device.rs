use crate::buffer::{AllocatedBuffer, VulkanBufferHandle};
use crate::context::VulkanContext;
use crate::error::GpuError;
use crate::sampled_image::VulkanSampledImage;
use ash::vk;
use gpu_allocator::MemoryLocation;
use infers_core::{
    CoreError, DataType, Deferred, Device, DeviceInfo, HardwareImage, HostTensor, Image,
    ImageFormat, Tensor, TensorAdopt, TensorShape,
};
use std::sync::Arc;

/// GPU execution device. Owns the shared [`VulkanContext`].
#[derive(Clone)]
pub struct Vulkan(Arc<VulkanContext>);

impl std::fmt::Debug for Vulkan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Vulkan")
            .field("info", self.info())
            .finish()
    }
}

impl Vulkan {
    pub fn new(gpu_id: usize) -> Result<Self, GpuError> {
        Ok(Self(Arc::new(VulkanContext::new_for_gpu(gpu_id)?)))
    }

    pub fn from_context(context: Arc<VulkanContext>) -> Self {
        Self(context)
    }

    pub fn context(&self) -> &Arc<VulkanContext> {
        &self.0
    }

    pub fn info(&self) -> &DeviceInfo {
        self.0.device_info()
    }

    pub fn same_context(&self, other: &Vulkan) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    pub fn ensure_same_context(&self, other: &Vulkan) -> Result<(), CoreError> {
        if self.same_context(other) {
            Ok(())
        } else {
            Err(CoreError::DeviceMismatch {
                expected: self.info().clone(),
                actual: other.info().clone(),
            })
        }
    }

    /// Reusable GPU image buffer for the SSBO preprocess path.
    pub fn image_buffer(
        &self,
        width: u32,
        height: u32,
        format: ImageFormat,
    ) -> Result<VulkanImageBuffer, CoreError> {
        VulkanImageBuffer::allocate(Arc::clone(&self.0), width, height, format)
            .map_err(CoreError::from)
    }
}

/// GPU tensor storage (`VkBuffer` on the owning context).
#[derive(Clone)]
pub struct VulkanStorage {
    inner: Arc<VulkanStorageInner>,
}

struct VulkanStorageInner {
    context: Arc<VulkanContext>,
    owned: Option<AllocatedBuffer>,
    external: Option<VulkanBufferHandle>,
}

impl std::fmt::Debug for VulkanStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanStorage").finish_non_exhaustive()
    }
}

impl VulkanStorage {
    pub fn context(&self) -> &Arc<VulkanContext> {
        &self.inner.context
    }

    pub fn vulkan_handle(&self) -> Option<VulkanBufferHandle> {
        if let Some(handle) = self.inner.external {
            Some(handle)
        } else {
            self.inner.owned.as_ref().map(|b| b.vulkan_handle())
        }
    }

    pub fn allocated_buffer(&self) -> Option<&AllocatedBuffer> {
        self.inner.owned.as_ref()
    }

    pub fn is_external(&self) -> bool {
        self.inner.external.is_some()
    }

    pub fn from_allocated(context: Arc<VulkanContext>, buffer: AllocatedBuffer) -> Self {
        Self {
            inner: Arc::new(VulkanStorageInner {
                context,
                owned: Some(buffer),
                external: None,
            }),
        }
    }

    /// Wrap a caller-owned buffer without taking ownership. The buffer is not destroyed on drop.
    pub fn from_external_handle(context: Arc<VulkanContext>, handle: VulkanBufferHandle) -> Self {
        Self {
            inner: Arc::new(VulkanStorageInner {
                context,
                owned: None,
                external: Some(handle),
            }),
        }
    }

    fn try_from_host_bytes(
        context: Arc<VulkanContext>,
        bytes: &[u8],
    ) -> Result<Self, GpuError> {
        let size = bytes.len() as u64;
        let mut staging = context.create_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            "upload-staging",
        )?;
        VulkanContext::write_allocation(&mut staging.allocation, bytes)?;

        let dst = context.create_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuOnly,
            "gpu-tensor-upload",
        )?;
        context.copy_buffer(staging.buffer, 0, dst.buffer, 0, size)?;
        context.destroy_buffer(staging);

        Ok(Self::from_allocated(context, dst))
    }

    fn try_allocate(context: Arc<VulkanContext>, byte_size: u64) -> Result<Self, GpuError> {
        let buffer = context.create_buffer(
            byte_size,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::CpuToGpu,
            "gpu-tensor",
        )?;
        Ok(Self::from_allocated(context, buffer))
    }

    fn try_read_bytes(&self) -> Result<Vec<u8>, GpuError> {
        let (buffer, offset, size) = self
            .inner
            .external
            .map(|h| (h.buffer, h.offset, h.size))
            .or_else(|| {
                self.inner.owned.as_ref().map(|src| {
                    (src.buffer, 0, src.size)
                })
            })
            .ok_or_else(|| GpuError::Other("GPU buffer already destroyed".into()))?;
        let staging = self.inner.context.create_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuToCpu,
            "readback-staging",
        )?;
        self.inner.context.copy_buffer(
            buffer,
            offset,
            staging.buffer,
            0,
            size,
        )?;
        let mut out = vec![0u8; size as usize];
        VulkanContext::read_allocation(&staging.allocation, &mut out)?;
        self.inner.context.destroy_buffer(staging);
        Ok(out)
    }
}

impl Drop for VulkanStorageInner {
    fn drop(&mut self) {
        if let Some(buffer) = self.owned.take() {
            self.context.destroy_buffer(buffer);
        }
    }
}

/// GPU-resident linear image (SSBO path) with reusable staging.
pub struct VulkanImageBuffer {
    context: Arc<VulkanContext>,
    width: u32,
    height: u32,
    format: ImageFormat,
    staging: Option<AllocatedBuffer>,
    gpu: Option<AllocatedBuffer>,
    /// Staging holds bytes not yet copied to [`Self::gpu`].
    staging_dirty: bool,
}

impl VulkanImageBuffer {
    fn allocate(
        context: Arc<VulkanContext>,
        width: u32,
        height: u32,
        format: ImageFormat,
    ) -> Result<Self, GpuError> {
        let byte_size = format.frame_bytes(width, height) as u64;
        let staging = context.create_buffer(
            byte_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            "image-staging",
        )?;
        let gpu = context.create_buffer(
            byte_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuOnly,
            "image-ssbo",
        )?;
        Ok(Self {
            context,
            width,
            height,
            format,
            staging: Some(staging),
            gpu: Some(gpu),
            staging_dirty: false,
        })
    }

    pub fn write_staging(&mut self, bytes: &[u8]) -> Result<(), CoreError> {
        let expected = self.format.frame_bytes(self.width, self.height) as usize;
        if bytes.len() != expected {
            return Err(CoreError::InvalidImageBuffer(format!(
                "expected {} bytes, got {}",
                expected,
                bytes.len()
            )));
        }
        let staging = self.staging.as_mut().ok_or_else(|| {
            CoreError::InvalidImageBuffer("image staging buffer destroyed".into())
        })?;
        VulkanContext::write_allocation(&mut staging.allocation, bytes)?;
        self.staging_dirty = true;
        Ok(())
    }

    /// Returns true when staged bytes still need copying to the GPU SSBO.
    pub fn staging_dirty(&self) -> bool {
        self.staging_dirty
    }

    /// Copy staged bytes to the GPU SSBO (separate from [`Self::write`] for deferred upload).
    pub fn flush_staging_to_gpu(&mut self) -> Result<(), CoreError> {
        let staging = self.staging.as_ref().ok_or_else(|| {
            CoreError::InvalidImageBuffer("image staging buffer destroyed".into())
        })?;
        let gpu = self.gpu.as_ref().ok_or_else(|| {
            CoreError::InvalidImageBuffer("image gpu buffer destroyed".into())
        })?;
        self.context
            .copy_buffer(staging.buffer, 0, gpu.buffer, 0, gpu.size)?;
        self.staging_dirty = false;
        Ok(())
    }

    pub fn write(&mut self, bytes: &[u8]) -> Result<(), CoreError> {
        self.write_staging(bytes)?;
        self.flush_staging_to_gpu()
    }

    pub fn gpu_buffer(&self) -> Option<&AllocatedBuffer> {
        self.gpu.as_ref()
    }

    pub fn staging_buffer(&self) -> Option<&AllocatedBuffer> {
        self.staging.as_ref()
    }

    pub fn context(&self) -> &Arc<VulkanContext> {
        &self.context
    }
}

impl Drop for VulkanImageBuffer {
    fn drop(&mut self) {
        if let Some(buf) = self.staging.take() {
            self.context.destroy_buffer(buf);
        }
        if let Some(buf) = self.gpu.take() {
            self.context.destroy_buffer(buf);
        }
    }
}

/// GPU image input: linear SSBO buffer or imported sampled image.
pub enum VulkanImage {
    Linear(VulkanImageBuffer),
    Sampled(Arc<VulkanSampledImage>),
}

impl VulkanImage {
    pub fn from_sampled(image: VulkanSampledImage) -> Self {
        Self::Sampled(Arc::new(image))
    }

    pub fn context(&self) -> &Arc<VulkanContext> {
        match self {
            Self::Linear(buf) => buf.context(),
            Self::Sampled(img) => img.vulkan_context(),
        }
    }

    pub fn width(&self) -> u32 {
        match self {
            Self::Linear(buf) => buf.width,
            Self::Sampled(img) => img.width(),
        }
    }

    pub fn height(&self) -> u32 {
        match self {
            Self::Linear(buf) => buf.height,
            Self::Sampled(img) => img.height(),
        }
    }

    pub fn format(&self) -> ImageFormat {
        match self {
            Self::Linear(buf) => buf.format,
            Self::Sampled(img) => img.format(),
        }
    }

    pub fn as_linear(&self) -> Option<&VulkanImageBuffer> {
        match self {
            Self::Linear(buf) => Some(buf),
            Self::Sampled(_) => None,
        }
    }

    pub fn as_sampled(&self) -> Option<&VulkanSampledImage> {
        match self {
            Self::Linear(_) => None,
            Self::Sampled(img) => Some(img),
        }
    }
}

impl Image<Vulkan> for VulkanImage {
    fn width(&self) -> u32 {
        self.width()
    }

    fn height(&self) -> u32 {
        self.height()
    }

    fn format(&self) -> ImageFormat {
        self.format()
    }
}

unsafe impl infers_core::sealed::Sealed for Vulkan {}

impl Device for Vulkan {
    type Storage = VulkanStorage;
    type Image = VulkanImage;

    fn info(&self) -> &DeviceInfo {
        self.info()
    }

    fn store(
        &self,
        shape: &TensorShape,
        dtype: DataType,
        bytes: &[u8],
    ) -> Result<Self::Storage, CoreError> {
        let expected = shape.byte_size(dtype);
        if bytes.len() != expected {
            return Err(CoreError::InvalidShape(format!(
                "Upload size mismatch: shape requires {} bytes, got {}",
                expected,
                bytes.len()
            )));
        }
        VulkanStorage::try_from_host_bytes(Arc::clone(&self.0), bytes).map_err(CoreError::from)
    }

    fn load(
        &self,
        storage: &Self::Storage,
        shape: &TensorShape,
        dtype: DataType,
    ) -> Result<HostTensor, CoreError> {
        if !Arc::ptr_eq(&self.0, storage.context()) {
            return Err(CoreError::DeviceMismatch {
                expected: self.info().clone(),
                actual: storage.context().device_info().clone(),
            });
        }
        let bytes = storage
            .try_read_bytes()
            .map_err(|e| CoreError::Gpu(Box::new(e)))?;
        HostTensor::new(shape.clone(), dtype, bytes)
    }

    fn materialize_deferred(
        device: &Self,
        hardware: std::sync::Arc<HardwareImage>,
    ) -> Result<Self::Image, CoreError> {
        let mut buffer =
            device.image_buffer(hardware.width(), hardware.height(), hardware.format())?;
        buffer.write_staging(hardware.as_bytes())?;
        Ok(VulkanImage::Linear(buffer))
    }
}

impl TensorAdopt for Vulkan {
    fn try_adopt_tensor<S: Device>(&self, tensor: &Tensor<S>) -> Option<Tensor<Self>> {
        if std::any::TypeId::of::<S>() != std::any::TypeId::of::<Vulkan>() {
            return None;
        }
        // SAFETY: S is Vulkan when TypeIds match.
        let vulkan_tensor = unsafe { &*(tensor as *const Tensor<S> as *const Tensor<Vulkan>) };
        if !Arc::ptr_eq(&self.0, vulkan_tensor.device().context()) {
            return None;
        }
        Some(Tensor::from_storage(
            self.clone(),
            vulkan_tensor.shape().clone(),
            vulkan_tensor.dtype(),
            vulkan_tensor.storage().clone(),
        ))
    }
}

/// Defer GPU placement of [`HardwareImage`] bytes without committing an upload yet.
pub fn defer_hardware(device: &Vulkan, hardware: HardwareImage) -> Deferred<Vulkan> {
    Deferred::from_hardware(device.clone(), std::sync::Arc::new(hardware))
}

/// Build a tensor from an already-allocated GPU buffer on `device`.
pub fn tensor_from_allocated(
    device: &Vulkan,
    shape: TensorShape,
    dtype: DataType,
    buffer: AllocatedBuffer,
) -> Tensor<Vulkan> {
    let storage = VulkanStorage::from_allocated(Arc::clone(device.context()), buffer);
    Tensor::from_storage(device.clone(), shape, dtype, storage)
}

/// Build a tensor view over a caller-owned external buffer on `device`.
pub fn tensor_from_external(
    device: &Vulkan,
    shape: TensorShape,
    dtype: DataType,
    handle: VulkanBufferHandle,
) -> Tensor<Vulkan> {
    let storage =
        VulkanStorage::from_external_handle(Arc::clone(device.context()), handle);
    Tensor::from_storage(device.clone(), shape, dtype, storage)
}
/// Allocate a zero-initialized GPU tensor on `device`.
pub fn allocate_tensor(
    device: &Vulkan,
    shape: TensorShape,
    dtype: DataType,
) -> Result<Tensor<Vulkan>, CoreError> {
    let bytes = shape.byte_size(dtype) as u64;
    let storage = VulkanStorage::try_allocate(Arc::clone(device.context()), bytes)
        .map_err(CoreError::from)?;
    Ok(Tensor::from_storage(device.clone(), shape, dtype, storage))
}
