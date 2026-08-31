use crate::device::Device;
use crate::error::CoreError;
use crate::tensor::{AnyHostTensor, DataType, TensorBuffer, TensorShape};

/// Performs real host↔device tensor transfers (e.g. via a shared [`VulkanContext`](infers_gpu::VulkanContext)).
///
/// Pass an implementor to [`TensorBuffer::copy_to_device`] when the target device
/// differs from the tensor's current residency.
pub trait DeviceTransfer: Send + Sync {
    /// Upload host tensor contents to a device-resident buffer.
    fn upload_tensor(
        &self,
        host: &dyn AnyHostTensor,
    ) -> Result<Box<dyn TensorBuffer>, CoreError>;

    /// Upload raw little-endian bytes to a device-resident buffer.
    fn upload_bytes(
        &self,
        device: &Device,
        shape: TensorShape,
        dtype: DataType,
        bytes: &[u8],
    ) -> Result<Box<dyn TensorBuffer>, CoreError>;
}
