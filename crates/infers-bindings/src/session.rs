use crate::error::InfersError;
use crate::tensor::{CpuTensor, TensorShape};
use infers_backend_executorch::ExecuTorchSession;
use infers_core::{Cpu, Session};
use parking_lot::Mutex;
use std::sync::Arc;

use crate::deferred::{CpuPending, GpuPending};

#[cfg(feature = "vulkan")]
use crate::tensor::GpuTensor;
#[cfg(feature = "vulkan")]
use infers_gpu::Vulkan;

#[derive(uniffi::Object)]
pub struct CpuSession {
    inner: Arc<Mutex<ExecuTorchSession<Cpu>>>,
    input_shapes: Vec<TensorShape>,
    output_shapes: Vec<TensorShape>,
}

impl std::fmt::Debug for CpuSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuSession")
            .field("input_shapes", &self.input_shapes)
            .field("output_shapes", &self.output_shapes)
            .finish()
    }
}

impl CpuSession {
    pub fn new(session: ExecuTorchSession<Cpu>) -> Self {
        let input_shapes = session
            .input_shapes()
            .iter()
            .cloned()
            .map(Into::into)
            .collect();
        let output_shapes = session
            .output_shapes()
            .iter()
            .cloned()
            .map(Into::into)
            .collect();

        Self {
            inner: Arc::new(Mutex::new(session)),
            input_shapes,
            output_shapes,
        }
    }
}

#[uniffi::export]
impl CpuSession {
    pub fn device_info(&self) -> crate::device::DeviceInfo {
        infers_core::Cpu::info().clone().into()
    }

    pub fn input_shapes(&self) -> Vec<TensorShape> {
        self.input_shapes.clone()
    }

    pub fn output_shapes(&self) -> Vec<TensorShape> {
        self.output_shapes.clone()
    }

    pub fn infer(
        &self,
        pending: Arc<CpuPending>,
    ) -> Result<Vec<Arc<CpuTensor>>, InfersError> {
        let mut lock = self.inner.lock();
        let outputs = lock.infer(pending.take()?).map_err(InfersError::from)?;
        Ok(outputs
            .into_iter()
            .map(|t| Arc::new(CpuTensor::from_inner(t)))
            .collect())
    }

    pub fn infer_tensor(
        &self,
        input: Arc<CpuTensor>,
    ) -> Result<Vec<Arc<CpuTensor>>, InfersError> {
        let mut lock = self.inner.lock();
        let outputs = lock.infer(input.inner()).map_err(InfersError::from)?;
        Ok(outputs
            .into_iter()
            .map(|t| Arc::new(CpuTensor::from_inner(t)))
            .collect())
    }
}

#[cfg(feature = "vulkan")]
#[derive(uniffi::Object)]
pub struct GpuSession {
    inner: Arc<Mutex<ExecuTorchSession<Vulkan>>>,
    input_shapes: Vec<TensorShape>,
    output_shapes: Vec<TensorShape>,
}

#[cfg(feature = "vulkan")]
impl std::fmt::Debug for GpuSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuSession")
            .field("input_shapes", &self.input_shapes)
            .field("output_shapes", &self.output_shapes)
            .finish()
    }
}

#[cfg(feature = "vulkan")]
impl GpuSession {
    pub fn new(session: ExecuTorchSession<Vulkan>) -> Self {
        let input_shapes = session
            .input_shapes()
            .iter()
            .cloned()
            .map(Into::into)
            .collect();
        let output_shapes = session
            .output_shapes()
            .iter()
            .cloned()
            .map(Into::into)
            .collect();

        Self {
            inner: Arc::new(Mutex::new(session)),
            input_shapes,
            output_shapes,
        }
    }
}

#[cfg(feature = "vulkan")]
#[uniffi::export]
impl GpuSession {
    pub fn device_info(&self) -> crate::device::DeviceInfo {
        self.inner.lock().device().info().clone().into()
    }

    pub fn input_shapes(&self) -> Vec<TensorShape> {
        self.input_shapes.clone()
    }

    pub fn output_shapes(&self) -> Vec<TensorShape> {
        self.output_shapes.clone()
    }

    pub fn infer(
        &self,
        pending: Arc<GpuPending>,
    ) -> Result<Vec<Arc<CpuTensor>>, InfersError> {
        let mut lock = self.inner.lock();
        let outputs = lock.infer(pending.take()?).map_err(InfersError::from)?;
        Ok(outputs
            .into_iter()
            .map(|t| Arc::new(CpuTensor::from_inner(t)))
            .collect())
    }

    pub fn infer_tensor(
        &self,
        input: Arc<GpuTensor>,
    ) -> Result<Vec<Arc<CpuTensor>>, InfersError> {
        let mut lock = self.inner.lock();
        let outputs = lock.infer(input.inner()).map_err(InfersError::from)?;
        Ok(outputs
            .into_iter()
            .map(|t| Arc::new(CpuTensor::from_inner(t)))
            .collect())
    }
}
