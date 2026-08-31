use std::sync::Arc;

use infers_core::{
    AnyHostTensor, Backend, CoreError, CpuImageBuffer, CpuTensor, DataType, Device, DeviceTransfer,
    FitMode, ImageFormat, ModelSession, ProcessingOptions, Rotation, SessionConfig, TensorBuffer,
    TensorLayout, TensorShape,
};

/// Simulated device-resident tensor (e.g. on GPU or NPU) for tests and examples.
#[derive(Clone, Debug)]
pub struct MockDeviceTensor {
    device: Device,
    shape: TensorShape,
    dtype: DataType,
    raw_payload: Vec<u8>,
}

impl MockDeviceTensor {
    pub fn from_f32_slice(device: Device, shape: TensorShape, values: &[f32]) -> Self {
        let mut raw_payload = Vec::with_capacity(values.len() * 4);
        for &v in values {
            raw_payload.extend_from_slice(&v.to_ne_bytes());
        }
        Self {
            device,
            shape,
            dtype: DataType::F32,
            raw_payload,
        }
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

    fn as_any(&self) -> &dyn std::any::Any {
        self
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

    fn copy_to_device(
        &self,
        target: &Device,
        _transfer: Option<&dyn DeviceTransfer>,
    ) -> Result<Box<dyn TensorBuffer>, CoreError> {
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
                let out_shape = TensorShape::new([1, 4]).expect("valid shape");
                let dummy_data = vec![10.0f32, 20.0, 100.0, 150.0];
                let out_tensor = MockDeviceTensor::from_f32_slice(dev.clone(), out_shape, &dummy_data);
                Ok(vec![Box::new(out_tensor) as Box<dyn TensorBuffer>])
            })
        });

        Ok(Box::new(MockSession::new(
            config.device.clone(),
            vec![TensorShape::new([1, 3, 224, 224]).expect("valid shape")],
            vec![TensorShape::new([1, 4]).expect("valid shape")],
            forward,
        )))
    }
}

/// Solid-color 640×480 RGB camera frame for pipeline tests.
pub fn camera_frame_640x480(fill: u8) -> CpuImageBuffer {
    let bytes = vec![fill; 640 * 480 * 3];
    CpuImageBuffer::new(640, 480, ImageFormat::Rgb888, bytes).expect("valid frame buffer")
}

pub fn detector_preprocess_options(dest: u32) -> ProcessingOptions {
    ProcessingOptions {
        src_w: 640,
        src_h: 480,
        crop_x: 0,
        crop_y: 0,
        crop_w: 640,
        crop_h: 480,
        dest_w: dest,
        dest_h: dest,
        src_format: ImageFormat::Rgb888,
        dest_format: ImageFormat::Rgbf32,
        fit_mode: FitMode::Contain,
        rotation: Rotation::None,
        dest_layout: TensorLayout::Nchw,
    }
}

pub fn landmarker_preprocess_options(
    box_x: f32,
    box_y: f32,
    box_w: f32,
    box_h: f32,
    dest: u32,
) -> ProcessingOptions {
    ProcessingOptions {
        src_w: 640,
        src_h: 480,
        crop_x: box_x.max(0.0) as u32,
        crop_y: box_y.max(0.0) as u32,
        crop_w: box_w.max(1.0) as u32,
        crop_h: box_h.max(1.0) as u32,
        dest_w: dest,
        dest_h: dest,
        src_format: ImageFormat::Rgb888,
        dest_format: ImageFormat::Rgbf32,
        fit_mode: FitMode::Stretch,
        rotation: Rotation::None,
        dest_layout: TensorLayout::Nchw,
    }
}

/// Mock GPU detector: one input `[1, 3, H, W]`, one output `[1, 4]` bounding box.
pub fn mock_gpu_detector(input_side: u32, device: Device) -> MockSession {
    let session_device = device.clone();
    let forward = Arc::new(move |_inputs: &[&dyn TensorBuffer]| {
        let shape = TensorShape::new([1, 4]).expect("valid shape");
        let boxes = vec![10.0f32, 20.0, 100.0, 120.0];
        Ok(vec![Box::new(MockDeviceTensor::from_f32_slice(
            session_device.clone(),
            shape,
            &boxes,
        )) as Box<dyn TensorBuffer>])
    });
    MockSession::new(
        device,
        vec![TensorShape::new([1, 3, input_side as usize, input_side as usize]).expect("valid shape")],
        vec![TensorShape::new([1, 4]).expect("valid shape")],
        forward,
    )
}

/// Mock GPU landmarker: one input `[1, 3, H, W]`, one output `[1, 4]` landmark pairs.
pub fn mock_gpu_landmarker(input_side: u32, device: Device) -> MockSession {
    let session_device = device.clone();
    let forward = Arc::new(move |_inputs: &[&dyn TensorBuffer]| {
        let shape = TensorShape::new([1, 4]).expect("valid shape");
        let points = vec![30.0f32, 40.0, 50.0, 60.0];
        Ok(vec![Box::new(MockDeviceTensor::from_f32_slice(
            session_device.clone(),
            shape,
            &points,
        )) as Box<dyn TensorBuffer>])
    });
    MockSession::new(
        device,
        vec![TensorShape::new([1, 3, input_side as usize, input_side as usize]).expect("valid shape")],
        vec![TensorShape::new([1, 4]).expect("valid shape")],
        forward,
    )
}

pub fn read_f32_output(outputs: &[Box<dyn TensorBuffer>]) -> Result<Vec<f32>, CoreError> {
    let host = outputs[0].read_to_cpu()?;
    Ok(host.as_slice_f32()?.to_vec())
}
