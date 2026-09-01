use crate::device::Device;
use crate::error::CoreError;
use crate::tensor::{Tensor, TensorShape};

pub trait Session<D: Device>: Send + Sync {
    /// Tensor this session produces. Backends whose delegate materializes
    /// results in host memory set this to `Tensor<Cpu>`.
    type Output;

    fn device(&self) -> &D;
    fn input_shapes(&self) -> &[TensorShape];
    fn output_shapes(&self) -> &[TensorShape];
    fn run(&mut self, inputs: &[&Tensor<D>]) -> Result<Vec<Self::Output>, CoreError>;
}
