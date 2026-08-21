use std::sync::Arc;
use crate::backend::{Backend, ModelSession, SessionConfig};
use crate::device::Device;
use crate::error::CoreError;
use crate::tensor::{AnyHostTensor, CpuTensor, DataType, TensorBuffer, TensorShape};

/// A simulated device-resident tensor (e.g. on GPU or NPU)
#[derive(Clone, Debug)]
pub struct MockDeviceTensor {
    device: Device,
    shape: TensorShape,
    dtype: DataType,
    // Raw backing payload kept opaque from direct CPU inspection
    raw_payload: Vec<u8>,
}

impl MockDeviceTensor {
    pub fn new(device: Device, shape: TensorShape, dtype: DataType, raw_payload: Vec<u8>) -> Self {
        Self {
            device,
            shape,
            dtype,
            raw_payload,
        }
    }

    pub fn from_f32_slice(device: Device, shape: TensorShape, values: &[f32]) -> Self {
        let mut raw_payload = Vec::with_capacity(values.len() * 4);
        for &v in values {
            raw_payload.extend_from_slice(&v.to_ne_bytes());
        }
        Self::new(device, shape, DataType::F32, raw_payload)
    }
}

impl TensorBuffer for MockDeviceTensor {
    fn shape(&self) -> &TensorShape {
        &self.shape
    }

    fn dtype(&self) -> DataType {
        self.dtype
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn read_to_cpu(&self) -> Result<Box<dyn AnyHostTensor>, CoreError> {
        match self.dtype {
            DataType::F32 => {
                let mut data = Vec::with_capacity(self.shape.element_count());
                for chunk in self.raw_payload.chunks_exact(4) {
                    let val = f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    data.push(val);
                }
                let tensor = CpuTensor::from_f32(self.shape.clone(), data)?;
                Ok(Box::new(tensor))
            }
            DataType::U8 => {
                let tensor = CpuTensor::from_u8(self.shape.clone(), self.raw_payload.clone())?;
                Ok(Box::new(tensor))
            }
            DataType::I32 => {
                let mut data = Vec::with_capacity(self.shape.element_count());
                for chunk in self.raw_payload.chunks_exact(4) {
                    let val = i32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    data.push(val);
                }
                let tensor = CpuTensor::from_i32(self.shape.clone(), data)?;
                Ok(Box::new(tensor))
            }
            _ => Err(CoreError::BufferTransferFailed(format!(
                "Readback for dtype {:?} not implemented in mock",
                self.dtype
            ))),
        }
    }

    fn copy_to_device(&self, target: &Device) -> Result<Box<dyn TensorBuffer>, CoreError> {
        if &self.device == target {
            Ok(Box::new(self.clone()))
        } else {
            Ok(Box::new(MockDeviceTensor {
                device: target.clone(),
                shape: self.shape.clone(),
                dtype: self.dtype,
                raw_payload: self.raw_payload.clone(),
            }))
        }
    }
}

pub type ForwardFn =
    Arc<dyn Fn(&[&dyn TensorBuffer]) -> Result<Vec<Box<dyn TensorBuffer>>, CoreError> + Send + Sync>;

pub struct MockSession {
    device: Device,
    input_shapes: Vec<TensorShape>,
    output_shapes: Vec<TensorShape>,
    forward: ForwardFn,
}

impl MockSession {
    pub fn new(
        device: Device,
        input_shapes: Vec<TensorShape>,
        output_shapes: Vec<TensorShape>,
        forward: ForwardFn,
    ) -> Self {
        Self {
            device,
            input_shapes,
            output_shapes,
            forward,
        }
    }
}

impl ModelSession for MockSession {
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
        // Enforce device safety
        for (i, input) in inputs.iter().enumerate() {
            if input.device() != &self.device {
                return Err(CoreError::DeviceMismatch {
                    expected: self.device.clone(),
                    actual: input.device().clone(),
                });
            }
            if i < self.input_shapes.len()
                && input.shape() != &self.input_shapes[i]
                && input.shape().element_count() != self.input_shapes[i].element_count()
            {
                return Err(CoreError::InvalidShape(format!(
                    "Input {} shape mismatch: expected {:?}, got {:?}",
                    i,
                    self.input_shapes[i].dims(),
                    input.shape().dims()
                )));
            }
        }

        (self.forward)(inputs)
    }
}

pub struct MockBackend {
    name: &'static str,
    devices: Vec<Device>,
    default_forward: Option<ForwardFn>,
}

impl MockBackend {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            devices: vec![Device::cpu(), Device::gpu(0), Device::npu(0)],
            default_forward: None,
        }
    }

    pub fn with_devices(mut self, devices: Vec<Device>) -> Self {
        self.devices = devices;
        self
    }

    pub fn with_default_forward(mut self, forward: ForwardFn) -> Self {
        self.default_forward = Some(forward);
        self
    }
}

impl Backend for MockBackend {
    fn name(&self) -> &'static str {
        self.name
    }

    fn available_devices(&self) -> Vec<Device> {
        self.devices.clone()
    }

    fn load_model_with_config(
        &self,
        _model_bytes: &[u8],
        config: &SessionConfig,
    ) -> Result<Box<dyn ModelSession>, CoreError> {
        let forward = self.default_forward.clone().unwrap_or_else(|| {
            let dev = config.device.clone();
            Arc::new(move |_inputs: &[&dyn TensorBuffer]| {
                let out_shape = TensorShape::from([1, 4]);
                let dummy_data = vec![10.0f32, 20.0, 100.0, 150.0];
                let out_tensor = MockDeviceTensor::from_f32_slice(dev.clone(), out_shape, &dummy_data);
                Ok(vec![Box::new(out_tensor) as Box<dyn TensorBuffer>])
            })
        });

        Ok(Box::new(MockSession::new(
            config.device.clone(),
            vec![TensorShape::from([1, 3, 224, 224])],
            vec![TensorShape::from([1, 4])],
            forward,
        )))
    }
}
