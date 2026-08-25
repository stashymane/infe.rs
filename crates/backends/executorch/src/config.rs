#[cfg(feature = "vulkan")]
use infers_gpu::VulkanContext;
#[cfg(feature = "vulkan")]
use std::sync::Arc;

/// Backend selection and parameters for [`crate::ExecuTorchBackend::load_model`].
#[derive(Clone)]
pub enum ExecuTorchBackendConfig {
    /// CPU execution via the XNNPACK (or portable) delegate path.
    Xnnpack {
        num_threads: usize,
        method: Option<String>,
    },
    /// GPU execution via the Vulkan delegate. Requires an existing shared context.
    ///
    /// Available when the `vulkan` crate feature is enabled (on by default).
    #[cfg(feature = "vulkan")]
    Vulkan {
        context: Arc<VulkanContext>,
        method: Option<String>,
    },
}

impl std::fmt::Debug for ExecuTorchBackendConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Xnnpack {
                num_threads,
                method,
            } => f
                .debug_struct("Xnnpack")
                .field("num_threads", num_threads)
                .field("method", method)
                .finish(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan { method, .. } => f
                .debug_struct("Vulkan")
                .field("method", method)
                .field("context", &"<VulkanContext>")
                .finish(),
        }
    }
}

impl ExecuTorchBackendConfig {
    pub fn method_name(&self) -> &str {
        match self {
            Self::Xnnpack { method, .. } => method.as_deref().unwrap_or("forward"),
            #[cfg(feature = "vulkan")]
            Self::Vulkan { method, .. } => method.as_deref().unwrap_or("forward"),
        }
    }
}
