use infers_core::DeviceKind;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ExecuTorchDelegate {
    Xnnpack,
    Vulkan,
    Qnn,
    CoreMl,
}

impl ExecuTorchDelegate {
    pub fn name(&self) -> &str {
        match self {
            ExecuTorchDelegate::Xnnpack => "xnnpack",
            ExecuTorchDelegate::Vulkan => "vulkan",
            ExecuTorchDelegate::Qnn => "qnn",
            ExecuTorchDelegate::CoreMl => "coreml",
        }
    }

    pub fn supports_device_kind(&self, kind: DeviceKind) -> bool {
        matches!(
            (self, kind),
            (ExecuTorchDelegate::Xnnpack, DeviceKind::Cpu)
                | (ExecuTorchDelegate::Vulkan, DeviceKind::Gpu)
                | (ExecuTorchDelegate::Qnn, DeviceKind::Npu)
                | (ExecuTorchDelegate::CoreMl, DeviceKind::Gpu | DeviceKind::Npu)
        )
    }
}
