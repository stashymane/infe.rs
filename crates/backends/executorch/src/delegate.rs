use infers_core::{Device, DeviceKind};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ExecuTorchDelegate {
    Xnnpack,
    Vulkan,
    Qnn,
    CoreMl,
    PortableCpu,
    Custom(String),
}

impl ExecuTorchDelegate {
    pub fn for_device(device: &Device) -> Self {
        match device.kind {
            DeviceKind::Cpu => ExecuTorchDelegate::Xnnpack,
            DeviceKind::Gpu => ExecuTorchDelegate::Vulkan,
            DeviceKind::Npu => ExecuTorchDelegate::Qnn,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            ExecuTorchDelegate::Xnnpack => "xnnpack",
            ExecuTorchDelegate::Vulkan => "vulkan",
            ExecuTorchDelegate::Qnn => "qnn",
            ExecuTorchDelegate::CoreMl => "coreml",
            ExecuTorchDelegate::PortableCpu => "portable_cpu",
            ExecuTorchDelegate::Custom(name) => name.as_str(),
        }
    }

    pub fn supports_device_kind(&self, kind: DeviceKind) -> bool {
        match (self, kind) {
            (ExecuTorchDelegate::Xnnpack | ExecuTorchDelegate::PortableCpu, DeviceKind::Cpu) => true,
            (ExecuTorchDelegate::Vulkan, DeviceKind::Gpu) => true,
            (ExecuTorchDelegate::Qnn, DeviceKind::Npu) => true,
            (ExecuTorchDelegate::CoreMl, DeviceKind::Gpu | DeviceKind::Npu) => true,
            (ExecuTorchDelegate::Custom(_), _) => true,
            _ => false,
        }
    }
}
