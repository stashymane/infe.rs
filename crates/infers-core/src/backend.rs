use crate::device::Device;
use crate::error::CoreError;
use crate::infer_input::InferInput;
use crate::pending::Pending;
use crate::tensor::{Tensor, TensorShape};

/// Session-side input preparation for [`Session::infer`].
pub trait SessionInputSink<D: Device> {
    fn materialize_pending(&mut self, pending: Pending<D>) -> Result<(), CoreError>;
    fn adopt_tensor(&mut self, tensor: &Tensor<D>) -> Result<(), CoreError>;
}

pub trait Session<D: Device>: Send + Sync {
    /// Tensor this session produces. Backends whose delegate materializes
    /// results in host memory set this to `Tensor<Cpu>`.
    type Output;

    fn device(&self) -> &D;
    fn input_shapes(&self) -> &[TensorShape];
    fn output_shapes(&self) -> &[TensorShape];

    /// Run inference. Accepts a [`Pending`] (commits preprocess into the session input)
    /// or a prepared [`Tensor`] reference.
    ///
    /// For GPU sessions, must be called on the **same thread** that materialized inputs
    /// (ExecuTorch Vulkan staging skip uses thread-local state).
    fn infer(&mut self, input: impl InferInput<D>) -> Result<Vec<Self::Output>, CoreError>;
}
