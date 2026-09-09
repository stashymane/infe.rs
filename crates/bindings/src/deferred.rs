use crate::device::DeviceInfo;
use crate::error::InfersError;
use crate::image::{
    CpuImage, CpuImageProcessor, GpuImage, GpuImageProcessor, HardwareImage, ProcessingOptions,
};
use crate::tensor::{CpuTensor, DataType, GpuTensor, TensorShape};
use infers_core::{Cpu, Deferred, Pending};
use parking_lot::Mutex;
use processing::{DeferredCpuProcessExt, DeferredVulkanProcessExt};
use processing_core::ProcessingOptions as CoreProcessingOptions;
use std::sync::Arc;

#[cfg(feature = "vulkan")]
use crate::gpu_device::GpuDevice;
#[cfg(feature = "vulkan")]
use infers_gpu::defer_hardware;

#[derive(uniffi::Object)]
pub struct CpuDeferred {
    inner: Mutex<Option<Deferred<Cpu>>>,
}

impl CpuDeferred {
    pub(crate) fn from_inner(inner: Deferred<Cpu>) -> Self {
        Self {
            inner: Mutex::new(Some(inner)),
        }
    }

    pub(crate) fn take(&self) -> Result<Deferred<Cpu>, InfersError> {
        self.inner.lock().take().ok_or(InfersError::AlreadyConsumed)
    }
}

#[uniffi::export]
impl CpuDeferred {
    pub fn device_info(&self) -> DeviceInfo {
        infers_core::Cpu::info().clone().into()
    }

    pub fn materialize(&self) -> Result<Arc<CpuImage>, InfersError> {
        let deferred = self.take()?;
        let image = deferred.materialize().map_err(InfersError::from)?;
        Ok(Arc::new(CpuImage::from_core(image)))
    }

    pub fn process(
        &self,
        processor: Arc<CpuImageProcessor>,
        options: ProcessingOptions,
    ) -> Result<Arc<CpuPending>, InfersError> {
        let deferred = self.take()?;
        let core_opts: CoreProcessingOptions = options.into();
        let pending = deferred
            .process(processor.core(), &core_opts)
            .map_err(InfersError::from)?;
        Ok(Arc::new(CpuPending::from_inner(pending)))
    }
}

#[derive(uniffi::Object)]
pub struct CpuPending {
    inner: Mutex<Option<Pending<Cpu>>>,
}

impl CpuPending {
    pub(crate) fn from_inner(inner: Pending<Cpu>) -> Self {
        Self {
            inner: Mutex::new(Some(inner)),
        }
    }

    pub(crate) fn take(&self) -> Result<Pending<Cpu>, InfersError> {
        self.inner.lock().take().ok_or(InfersError::AlreadyConsumed)
    }

    fn inner(&self) -> Result<PendingGuard<'_, Cpu>, InfersError> {
        PendingGuard::new(&self.inner)
    }
}

struct PendingGuard<'a, D: infers_core::Device> {
    slot: parking_lot::MutexGuard<'a, Option<Pending<D>>>,
}

impl<'a, D: infers_core::Device> PendingGuard<'a, D> {
    fn new(mutex: &'a Mutex<Option<Pending<D>>>) -> Result<Self, InfersError> {
        let slot = mutex.lock();
        if slot.is_none() {
            return Err(InfersError::AlreadyConsumed);
        }
        Ok(Self { slot })
    }

    fn get(&self) -> &Pending<D> {
        self.slot.as_ref().expect("checked above")
    }
}

#[uniffi::export]
impl CpuPending {
    pub fn shape(&self) -> Result<TensorShape, InfersError> {
        Ok(self.inner()?.get().shape().clone().into())
    }

    pub fn dtype(&self) -> Result<DataType, InfersError> {
        Ok(self.inner()?.get().dtype().into())
    }

    pub fn device_info(&self) -> DeviceInfo {
        infers_core::Cpu::info().clone().into()
    }

    pub fn materialize(&self) -> Result<Arc<CpuTensor>, InfersError> {
        let pending = self.take()?;
        let tensor = pending.materialize().map_err(InfersError::from)?;
        Ok(Arc::new(CpuTensor::from_inner(tensor)))
    }
}

#[cfg(feature = "vulkan")]
#[derive(uniffi::Object)]
pub struct GpuDeferred {
    inner: Mutex<Option<Deferred<infers_gpu::Vulkan>>>,
}

#[cfg(feature = "vulkan")]
impl GpuDeferred {
    pub(crate) fn from_inner(inner: Deferred<infers_gpu::Vulkan>) -> Self {
        Self {
            inner: Mutex::new(Some(inner)),
        }
    }

    pub(crate) fn take(&self) -> Result<Deferred<infers_gpu::Vulkan>, InfersError> {
        self.inner.lock().take().ok_or(InfersError::AlreadyConsumed)
    }
}

#[cfg(feature = "vulkan")]
#[uniffi::export]
impl GpuDeferred {
    pub fn device_info(&self) -> Result<DeviceInfo, InfersError> {
        Ok(self
            .inner
            .lock()
            .as_ref()
            .ok_or(InfersError::AlreadyConsumed)?
            .device()
            .info()
            .clone()
            .into())
    }

    pub fn materialize(&self) -> Result<Arc<GpuImage>, InfersError> {
        let deferred = self.take()?;
        let image = deferred.materialize().map_err(InfersError::from)?;
        Ok(Arc::new(GpuImage::from_vulkan(image)))
    }

    pub fn process(
        &self,
        processor: Arc<GpuImageProcessor>,
        options: ProcessingOptions,
    ) -> Result<Arc<GpuPending>, InfersError> {
        let deferred = self.take()?;
        let core_opts: CoreProcessingOptions = options.into();
        let pending = deferred
            .process(processor.core(), &core_opts)
            .map_err(InfersError::from)?;
        Ok(Arc::new(GpuPending::from_inner(pending)))
    }
}

#[cfg(feature = "vulkan")]
#[derive(uniffi::Object)]
pub struct GpuPending {
    inner: Mutex<Option<Pending<infers_gpu::Vulkan>>>,
}

#[cfg(feature = "vulkan")]
impl GpuPending {
    pub(crate) fn from_inner(inner: Pending<infers_gpu::Vulkan>) -> Self {
        Self {
            inner: Mutex::new(Some(inner)),
        }
    }

    pub(crate) fn take(&self) -> Result<Pending<infers_gpu::Vulkan>, InfersError> {
        self.inner.lock().take().ok_or(InfersError::AlreadyConsumed)
    }

    fn inner(&self) -> Result<PendingGuard<'_, infers_gpu::Vulkan>, InfersError> {
        PendingGuard::new(&self.inner)
    }
}

#[cfg(feature = "vulkan")]
#[uniffi::export]
impl GpuPending {
    pub fn shape(&self) -> Result<TensorShape, InfersError> {
        Ok(self.inner()?.get().shape().clone().into())
    }

    pub fn dtype(&self) -> Result<DataType, InfersError> {
        Ok(self.inner()?.get().dtype().into())
    }

    pub fn device_info(&self) -> Result<DeviceInfo, InfersError> {
        Ok(self
            .inner()?
            .get()
            .device()
            .info()
            .clone()
            .into())
    }

    pub fn materialize(&self) -> Result<Arc<GpuTensor>, InfersError> {
        let pending = self.take()?;
        let tensor = pending.materialize().map_err(InfersError::from)?;
        Ok(Arc::new(GpuTensor::from_inner(tensor)))
    }
}

#[uniffi::export]
impl HardwareImage {
    pub fn on_cpu(self: Arc<Self>) -> Arc<CpuDeferred> {
        let hardware = Arc::new(self.inner().clone());
        Arc::new(CpuDeferred::from_inner(Deferred::from_hardware(Cpu, hardware)))
    }

    #[cfg(feature = "vulkan")]
    pub fn on(self: Arc<Self>, device: Arc<GpuDevice>) -> Arc<GpuDeferred> {
        let hardware = self.inner().clone();
        Arc::new(GpuDeferred::from_inner(defer_hardware(
            device.vulkan(),
            hardware,
        )))
    }
}
