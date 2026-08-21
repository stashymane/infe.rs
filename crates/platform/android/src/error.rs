use infers_core::CoreError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AndroidPlatformError {
    #[error("AHardwareBuffer allocation failed with error code {0}")]
    AllocationFailed(i32),

    #[error("AHardwareBuffer lock failed with error code {0}")]
    LockFailed(i32),

    #[error("AHardwareBuffer unlock failed with error code {0}")]
    UnlockFailed(i32),

    #[error("Unsupported hardware buffer format: {0}")]
    UnsupportedFormat(u32),

    #[error("Null buffer pointer encountered")]
    NullBufferPointer,

    #[error("Vulkan external memory import error: {0}")]
    VulkanImportError(String),

    #[error("JNI error: {0}")]
    JniError(String),

    #[error("Core error: {0}")]
    Core(#[from] CoreError),
}
