use crate::config::XnnpackOptions;
#[cfg(feature = "vulkan")]
use crate::config::VulkanOptions;
use crate::session::ExecuTorchSession;
use infers_core::{CoreError, Cpu};
#[cfg(feature = "vulkan")]
use infers_gpu::Vulkan;

/// The ExecuTorch inference backend.
#[derive(Clone, Debug, Default)]
pub struct ExecuTorchBackend;

impl ExecuTorchBackend {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    pub fn load_xnnpack(
        &self,
        model_bytes: &[u8],
        opts: XnnpackOptions,
    ) -> Result<ExecuTorchSession<Cpu>, CoreError> {
        ExecuTorchSession::load_xnnpack(model_bytes, &opts).map_err(CoreError::from)
    }

    pub fn load_xnnpack_from_file(
        &self,
        path: &str,
        opts: XnnpackOptions,
    ) -> Result<ExecuTorchSession<Cpu>, CoreError> {
        ExecuTorchSession::load_xnnpack_from_path(std::path::Path::new(path), &opts)
            .map_err(CoreError::from)
    }

    #[cfg(feature = "vulkan")]
    pub fn load_vulkan(
        &self,
        model_bytes: &[u8],
        device: &Vulkan,
        opts: VulkanOptions,
    ) -> Result<ExecuTorchSession<Vulkan>, CoreError> {
        ExecuTorchSession::load_vulkan(model_bytes, device, &opts).map_err(CoreError::from)
    }

    #[cfg(feature = "vulkan")]
    pub fn load_vulkan_from_file(
        &self,
        path: &str,
        device: &Vulkan,
        opts: VulkanOptions,
    ) -> Result<ExecuTorchSession<Vulkan>, CoreError> {
        ExecuTorchSession::load_vulkan_from_path(std::path::Path::new(path), device, &opts)
            .map_err(CoreError::from)
    }
}
