#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, uniffi::Enum)]
pub enum DeviceKind {
    Cpu,
    Gpu,
    Npu,
}

impl From<infers_core::DeviceKind> for DeviceKind {
    fn from(kind: infers_core::DeviceKind) -> Self {
        match kind {
            infers_core::DeviceKind::Cpu => DeviceKind::Cpu,
            infers_core::DeviceKind::Gpu => DeviceKind::Gpu,
            infers_core::DeviceKind::Npu => DeviceKind::Npu,
        }
    }
}

impl From<DeviceKind> for infers_core::DeviceKind {
    fn from(kind: DeviceKind) -> Self {
        match kind {
            DeviceKind::Cpu => infers_core::DeviceKind::Cpu,
            DeviceKind::Gpu => infers_core::DeviceKind::Gpu,
            DeviceKind::Npu => infers_core::DeviceKind::Npu,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, uniffi::Record)]
pub struct Device {
    pub kind: DeviceKind,
    pub id: u64,
    pub name: String,
}

impl From<infers_core::Device> for Device {
    fn from(device: infers_core::Device) -> Self {
        Self {
            kind: device.kind.into(),
            id: device.id as u64,
            name: device.name,
        }
    }
}

impl From<Device> for infers_core::Device {
    fn from(device: Device) -> Self {
        Self {
            kind: device.kind.into(),
            id: device.id as usize,
            name: device.name,
        }
    }
}

#[uniffi::export]
pub fn create_cpu_device() -> Device {
    infers_core::Device::cpu().into()
}

#[uniffi::export]
pub fn create_gpu_device(id: u64) -> Device {
    infers_core::Device::gpu(id as usize).into()
}

#[uniffi::export]
pub fn create_npu_device(id: u64) -> Device {
    infers_core::Device::npu(id as usize).into()
}
