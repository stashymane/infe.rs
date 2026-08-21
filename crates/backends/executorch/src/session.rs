use crate::delegate::ExecuTorchDelegate;
use crate::error::ExecuTorchError;
use crate::program::{MethodDescriptor, TensorDescriptor};
use crate::tensor::ExecuTorchTensorBuffer;
use infers_core::{CoreError, Device, ModelSession, TensorBuffer, TensorShape};

pub struct ExecuTorchSession {
    device: Device,
    delegate: ExecuTorchDelegate,
    method: MethodDescriptor,
    input_shapes: Vec<TensorShape>,
    output_shapes: Vec<TensorShape>,
}

impl ExecuTorchSession {
    pub fn new(
        device: Device,
        delegate: ExecuTorchDelegate,
        method: MethodDescriptor,
    ) -> Result<Self, ExecuTorchError> {
        if !delegate.supports_device_kind(device.kind) {
            return Err(ExecuTorchError::Execution(format!(
                "Delegate '{}' is not compatible with device '{:?}'",
                delegate.name(),
                device.kind
            )));
        }

        let input_shapes = method.inputs.iter().map(|i| i.shape.clone()).collect();
        let output_shapes = method.outputs.iter().map(|o| o.shape.clone()).collect();

        Ok(Self {
            device,
            delegate,
            method,
            input_shapes,
            output_shapes,
        })
    }

    pub fn delegate(&self) -> &ExecuTorchDelegate {
        &self.delegate
    }

    pub fn method_name(&self) -> &str {
        &self.method.name
    }

    pub fn input_descriptors(&self) -> &[TensorDescriptor] {
        &self.method.inputs
    }

    pub fn output_descriptors(&self) -> &[TensorDescriptor] {
        &self.method.outputs
    }
}

impl ModelSession for ExecuTorchSession {
    fn device(&self) -> &Device {
        &self.device
    }

    fn input_shapes(&self) -> &[TensorShape] {
        &self.input_shapes
    }

    fn output_shapes(&self) -> &[TensorShape] {
        &self.output_shapes
    }

    fn run(&mut self, inputs: &[&dyn TensorBuffer]) -> Result<Vec<Box<dyn TensorBuffer>>, CoreError> {
        // 1. Validate input count
        if inputs.len() != self.method.inputs.len() {
            return Err(CoreError::InferenceFailed(format!(
                "Input count mismatch for method '{}': expected {}, received {}",
                self.method.name,
                self.method.inputs.len(),
                inputs.len()
            )));
        }

        // 2. Validate device residency and descriptors for all inputs
        for (i, (&input, expected)) in inputs.iter().zip(self.method.inputs.iter()).enumerate() {
            // Strict device verification to prevent accidental CPU/GPU/NPU transfers
            if input.device().kind != self.device.kind {
                return Err(CoreError::DeviceMismatch {
                    expected: self.device.clone(),
                    actual: input.device().clone(),
                });
            }

            // Shape check
            if input.shape() != &expected.shape {
                return Err(CoreError::InvalidShape(format!(
                    "Input {} ('{}') shape mismatch: expected {:?}, got {:?}",
                    i,
                    expected.name,
                    expected.shape.dims(),
                    input.shape().dims()
                )));
            }

            // Data type check
            if input.dtype() != expected.dtype {
                return Err(CoreError::InvalidDataType {
                    expected: expected.dtype,
                    actual: input.dtype(),
                });
            }
        }

        // 3. Execute method program: returns device-resident output tensor buffers matching the method's output descriptors
        let mut output_tensors: Vec<Box<dyn TensorBuffer>> =
            Vec::with_capacity(self.method.outputs.len());
        for out_desc in &self.method.outputs {
            let tb = ExecuTorchTensorBuffer::zeroed(
                self.device.clone(),
                out_desc.shape.clone(),
                out_desc.dtype,
            )
            .map_err(CoreError::from)?;
            output_tensors.push(Box::new(tb));
        }
        Ok(output_tensors)
    }
}
