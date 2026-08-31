use crate::gpu::buffer::GpuTensorBuffer;
use infers_core::{
    AnyHostTensor, CoreError, DataType, Device, DeviceTransfer, TensorBuffer, TensorShape,
};
use infers_gpu::VulkanContext;
use std::sync::Arc;

/// Upload any host-resident [`TensorBuffer`] to GPU memory.
pub fn upload_tensor_buffer(
    context: &Arc<VulkanContext>,
    input: &dyn TensorBuffer,
) -> Result<GpuTensorBuffer, CoreError> {
    if let Some(gpu) = input.as_any().downcast_ref::<GpuTensorBuffer>() {
        if gpu.device() == context.logical_device() {
            return Ok(gpu.clone());
        }
        return Err(CoreError::BufferTransferFailed(
            "Cross-GPU tensor upload is not supported".into(),
        ));
    }
    let host = input.read_to_cpu()?;
    GpuTensorBuffer::from_host_bytes(
        Arc::clone(context),
        input.shape().clone(),
        input.dtype(),
        host.as_bytes(),
    )
}

/// [`DeviceTransfer`] adapter that holds the shared Vulkan context.
pub struct VulkanDeviceTransfer {
    context: Arc<VulkanContext>,
}

impl VulkanDeviceTransfer {
    pub fn new(context: Arc<VulkanContext>) -> Self {
        Self { context }
    }

    pub fn context(&self) -> &Arc<VulkanContext> {
        &self.context
    }
}

impl DeviceTransfer for VulkanDeviceTransfer {
    fn upload_tensor(
        &self,
        host: &dyn AnyHostTensor,
    ) -> Result<Box<dyn TensorBuffer>, CoreError> {
        let tensor = GpuTensorBuffer::from_host_bytes(
            Arc::clone(&self.context),
            host.shape().clone(),
            host.dtype(),
            host.as_bytes(),
        )?;
        Ok(Box::new(tensor))
    }

    fn upload_bytes(
        &self,
        device: &Device,
        shape: TensorShape,
        dtype: DataType,
        bytes: &[u8],
    ) -> Result<Box<dyn TensorBuffer>, CoreError> {
        if device != self.context.logical_device() {
            return Err(CoreError::BufferTransferFailed(format!(
                "Upload target {} does not match VulkanContext device {}",
                device,
                self.context.logical_device()
            )));
        }
        let tensor = GpuTensorBuffer::from_host_bytes(
            Arc::clone(&self.context),
            shape,
            dtype,
            bytes,
        )?;
        Ok(Box::new(tensor))
    }
}
