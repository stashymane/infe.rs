use crate::device::Device;
use crate::error::InfersError;
use crate::tensor::{TensorBuffer, TensorShape};
use infers_core::ModelSession as CoreModelSession;
use parking_lot::Mutex;
use std::sync::Arc;

#[derive(uniffi::Object)]
pub struct ModelSession {
    inner: Arc<Mutex<Box<dyn CoreModelSession>>>,
    device: Device,
    input_shapes: Vec<TensorShape>,
    output_shapes: Vec<TensorShape>,
}

impl std::fmt::Debug for ModelSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelSession")
            .field("device", &self.device)
            .field("input_shapes", &self.input_shapes)
            .field("output_shapes", &self.output_shapes)
            .finish()
    }
}

impl ModelSession {
    pub fn new(session: Box<dyn CoreModelSession>) -> Self {
        let device = session.device().clone().into();
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
            device,
            input_shapes,
            output_shapes,
        }
    }
}

#[uniffi::export]
impl ModelSession {
    pub fn device(&self) -> Device {
        self.device.clone()
    }

    pub fn input_shapes(&self) -> Vec<TensorShape> {
        self.input_shapes.clone()
    }

    pub fn output_shapes(&self) -> Vec<TensorShape> {
        self.output_shapes.clone()
    }

    pub fn run(&self, inputs: Vec<Arc<TensorBuffer>>) -> Result<Vec<Arc<TensorBuffer>>, InfersError> {
        let core_inputs: Vec<&dyn infers_core::TensorBuffer> =
            inputs.iter().map(|tb| tb.as_core()).collect();

        let mut lock = self.inner.lock();
        let outputs = lock.run(&core_inputs).map_err(InfersError::from)?;

        let wrapped = outputs
            .into_iter()
            .map(|buf| Arc::new(TensorBuffer::from_boxed(buf)))
            .collect();

        Ok(wrapped)
    }
}
