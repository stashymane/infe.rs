use crate::backend::SessionInputSink;
use crate::device::Device;
use crate::error::CoreError;
use crate::pending::Pending;
use crate::tensor::Tensor;

/// Input accepted by [`crate::Session::infer`]: a deferred pending commit or prepared tensor borrow.
pub trait InferInput<D: Device> {
    fn prepare(self, session: &mut dyn SessionInputSink<D>) -> Result<(), CoreError>;
}

impl<D: Device> InferInput<D> for Pending<D> {
    fn prepare(self, session: &mut dyn SessionInputSink<D>) -> Result<(), CoreError> {
        session.materialize_pending(self)
    }
}

impl<'a, D: Device> InferInput<D> for &'a Tensor<D> {
    fn prepare(self, session: &mut dyn SessionInputSink<D>) -> Result<(), CoreError> {
        session.adopt_tensor(self)
    }
}

/// Prepare a single session input from a pending commit or prepared tensor.
pub fn prepare_infer<D: Device, I: InferInput<D>>(
    input: I,
    sink: &mut dyn SessionInputSink<D>,
) -> Result<(), CoreError> {
    input.prepare(sink)
}
