#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    Cpu,
    Gpu,
    Npu,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Device {
    pub kind: DeviceKind,
    pub id: usize,
    pub name: String,
}

impl Device {
    #[inline]
    #[must_use]
    pub fn cpu() -> Self {
        Self {
            kind: DeviceKind::Cpu,
            id: 0,
            name: "CPU".to_string(),
        }
    }

    #[inline]
    #[must_use]
    pub fn gpu(id: usize) -> Self {
        Self {
            kind: DeviceKind::Gpu,
            id,
            name: format!("GPU:{}", id),
        }
    }

    #[inline]
    #[must_use]
    pub fn npu(id: usize) -> Self {
        Self {
            kind: DeviceKind::Npu,
            id,
            name: format!("NPU:{}", id),
        }
    }

    #[inline]
    pub fn is_cpu(&self) -> bool {
        self.kind == DeviceKind::Cpu
    }

    #[inline]
    pub fn is_gpu(&self) -> bool {
        self.kind == DeviceKind::Gpu
    }

    #[inline]
    pub fn is_npu(&self) -> bool {
        self.kind == DeviceKind::Npu
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

static CPU_DEVICE: std::sync::OnceLock<Device> = std::sync::OnceLock::new();

#[inline]
pub(crate) fn cpu_device() -> &'static Device {
    CPU_DEVICE.get_or_init(Device::cpu)
}
