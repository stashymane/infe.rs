#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, uniffi::Enum)]
pub enum DeviceKind {
    Cpu,
    Gpu,
    Npu,
}

uniffi_mirror! {
    DeviceKind <=> infers_core::DeviceKind,
    [Cpu, Gpu, Npu]
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, uniffi::Record)]
pub struct DeviceInfo {
    pub kind: DeviceKind,
    pub id: u64,
    pub name: String,
}

impl From<infers_core::DeviceInfo> for DeviceInfo {
    fn from(info: infers_core::DeviceInfo) -> Self {
        Self {
            kind: info.kind.into(),
            id: info.id as u64,
            name: info.name,
        }
    }
}

impl From<DeviceInfo> for infers_core::DeviceInfo {
    fn from(info: DeviceInfo) -> Self {
        Self {
            kind: info.kind.into(),
            id: info.id as usize,
            name: info.name,
        }
    }
}

/// CPU execution device (singleton metadata).
#[derive(uniffi::Object)]
pub struct CpuDevice;

#[uniffi::export]
impl CpuDevice {
    #[uniffi::constructor]
    pub fn new() -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self)
    }

    pub fn info(&self) -> DeviceInfo {
        infers_core::Cpu::info().clone().into()
    }
}
