pub mod assets;

use std::sync::Arc;

use infers_core::{
    CoreError, Cpu, FitMode, HardwareImage, HostTensor, ImageFormat, InferInput, Pending,
    ProcessingOptions, Session, SessionInputSink, Tensor, TensorLayout, TensorShape,
    prepare_infer,
};

#[cfg(feature = "vulkan")]
use infers_gpu::Vulkan;

pub type CpuForwardFn =
    Arc<dyn Fn(&[&Tensor<Cpu>]) -> Result<Vec<Tensor<Cpu>>, CoreError> + Send + Sync>;

pub struct MockCpuSession {
    input_shapes: Vec<TensorShape>,
    output_shapes: Vec<TensorShape>,
    forward: CpuForwardFn,
}

impl MockCpuSession {
    pub fn new(
        input_shapes: Vec<TensorShape>,
        output_shapes: Vec<TensorShape>,
        forward: CpuForwardFn,
    ) -> Self {
        Self {
            input_shapes,
            output_shapes,
            forward,
        }
    }
}

impl Session<Cpu> for MockCpuSession {
    type Output = Tensor<Cpu>;

    fn device(&self) -> &Cpu {
        &Cpu
    }

    fn input_shapes(&self) -> &[TensorShape] {
        &self.input_shapes
    }

    fn output_shapes(&self) -> &[TensorShape] {
        &self.output_shapes
    }

    fn infer(&mut self, input: impl InferInput<Cpu>) -> Result<Vec<Tensor<Cpu>>, CoreError> {
        let mut sink = SingleInputSink::default();
        let tensor = sink.prepare(input)?;
        if let Some(expected) = self.input_shapes.first() {
            if tensor.shape() != expected && tensor.shape().element_count() != expected.element_count()
            {
                return Err(CoreError::InvalidShape(format!(
                    "Input shape mismatch: expected {:?}, got {:?}",
                    expected.dims(),
                    tensor.shape().dims()
                )));
            }
        }
        (self.forward)(&[&tensor])
    }
}

#[cfg(feature = "vulkan")]
pub type VulkanForwardFn =
    Arc<dyn Fn(&[&Tensor<Vulkan>]) -> Result<Vec<Tensor<Cpu>>, CoreError> + Send + Sync>;

#[cfg(feature = "vulkan")]
pub struct MockGpuSession {
    device: Vulkan,
    input_shapes: Vec<TensorShape>,
    output_shapes: Vec<TensorShape>,
    forward: VulkanForwardFn,
}

#[cfg(feature = "vulkan")]
impl MockGpuSession {
    pub fn new(
        device: Vulkan,
        input_shapes: Vec<TensorShape>,
        output_shapes: Vec<TensorShape>,
        forward: VulkanForwardFn,
    ) -> Self {
        Self {
            device,
            input_shapes,
            output_shapes,
            forward,
        }
    }
}

#[cfg(feature = "vulkan")]
impl Session<Vulkan> for MockGpuSession {
    type Output = Tensor<Cpu>;

    fn device(&self) -> &Vulkan {
        &self.device
    }

    fn input_shapes(&self) -> &[TensorShape] {
        &self.input_shapes
    }

    fn output_shapes(&self) -> &[TensorShape] {
        &self.output_shapes
    }

    fn infer(&mut self, input: impl InferInput<Vulkan>) -> Result<Vec<Tensor<Cpu>>, CoreError> {
        let mut sink = SingleInputSinkVulkan::new();
        let tensor = sink.prepare(input)?;
        if !Arc::ptr_eq(tensor.device().context(), self.device.context()) {
            return Err(CoreError::DeviceMismatch {
                expected: self.device.info().clone(),
                actual: tensor.device().info().clone(),
            });
        }
        if let Some(expected) = self.input_shapes.first() {
            if tensor.shape() != expected && tensor.shape().element_count() != expected.element_count()
            {
                return Err(CoreError::InvalidShape(format!(
                    "Input shape mismatch: expected {:?}, got {:?}",
                    expected.dims(),
                    tensor.shape().dims()
                )));
            }
        }
        (self.forward)(&[&tensor])
    }
}

struct SingleInputSink {
    tensor: Option<Tensor<Cpu>>,
}

impl Default for SingleInputSink {
    fn default() -> Self {
        Self { tensor: None }
    }
}

impl SingleInputSink {
    fn prepare(&mut self, input: impl InferInput<Cpu>) -> Result<Tensor<Cpu>, CoreError> {
        prepare_infer(input, self)?;
        self.tensor
            .take()
            .ok_or_else(|| CoreError::InferenceFailed("missing inference input".into()))
    }
}

#[cfg(feature = "vulkan")]
struct SingleInputSinkVulkan {
    tensor: Option<Tensor<Vulkan>>,
}

#[cfg(feature = "vulkan")]
impl SingleInputSinkVulkan {
    fn new() -> Self {
        Self { tensor: None }
    }

    fn prepare(&mut self, input: impl InferInput<Vulkan>) -> Result<Tensor<Vulkan>, CoreError> {
        prepare_infer(input, self)?;
        self.tensor
            .take()
            .ok_or_else(|| CoreError::InferenceFailed("missing inference input".into()))
    }
}

impl SessionInputSink<Cpu> for SingleInputSink {
    fn materialize_pending(&mut self, pending: Pending<Cpu>) -> Result<(), CoreError> {
        self.tensor = Some(pending.materialize()?);
        Ok(())
    }

    fn adopt_tensor(&mut self, tensor: &Tensor<Cpu>) -> Result<(), CoreError> {
        self.tensor = Some(tensor.clone());
        Ok(())
    }
}

#[cfg(feature = "vulkan")]
impl SessionInputSink<Vulkan> for SingleInputSinkVulkan {
    fn materialize_pending(&mut self, pending: Pending<Vulkan>) -> Result<(), CoreError> {
        self.tensor = Some(pending.materialize()?);
        Ok(())
    }

    fn adopt_tensor(&mut self, tensor: &Tensor<Vulkan>) -> Result<(), CoreError> {
        self.tensor = Some(tensor.clone());
        Ok(())
    }
}

pub fn cpu_tensor_f32(shape: TensorShape, values: &[f32]) -> Result<Tensor<Cpu>, CoreError> {
    let host = HostTensor::from_f32(shape, values.to_vec())?;
    Tensor::from_host(&Cpu, &host)
}

pub fn read_f32_output(outputs: &[Tensor<Cpu>]) -> Result<Vec<f32>, CoreError> {
    Ok(outputs[0].read_to_host()?.as_slice_f32()?.to_vec())
}

#[cfg(feature = "vulkan")]
pub fn gpu_tensor_f32(device: &Vulkan, shape: TensorShape, values: &[f32]) -> Result<Tensor<Vulkan>, CoreError> {
    let cpu = cpu_tensor_f32(shape, values)?;
    cpu.to_device(device)
}

/// Solid-color 640×480 RGB camera frame for pipeline tests.
pub fn camera_frame_640x480(fill: u8) -> HardwareImage {
    let bytes = vec![fill; 640 * 480 * 3];
    HardwareImage::new(640, 480, ImageFormat::Rgb888, bytes).expect("valid frame buffer")
}

/// Synthetic frame plus every bundled sample JPEG under `assets/images/`.
pub fn test_input_frames() -> Vec<(&'static str, HardwareImage)> {
    let mut frames = vec![("synthetic-640x480", camera_frame_640x480(128))];
    for sample in assets::SampleImage::ALL {
        let image = assets::load_sample_image(*sample)
            .unwrap_or_else(|err| panic!("load sample {}: {err}", sample.label()));
        frames.push((sample.label(), image));
    }
    frames
}

pub fn detector_preprocess_options(frame: &HardwareImage, dest: u32) -> ProcessingOptions {
    ProcessingOptions {
        src_w: frame.width(),
        src_h: frame.height(),
        crop_x: 0,
        crop_y: 0,
        crop_w: frame.width(),
        crop_h: frame.height(),
        dest_w: dest,
        dest_h: dest,
        src_format: frame.format(),
        dest_format: ImageFormat::Rgbf32,
        fit_mode: FitMode::Contain,
        rotation_degrees: 0.0,
        dest_layout: TensorLayout::Nchw,
    }
}

pub fn landmarker_preprocess_options(
    frame: &HardwareImage,
    box_x: f32,
    box_y: f32,
    box_w: f32,
    box_h: f32,
    dest: u32,
) -> ProcessingOptions {
    ProcessingOptions {
        src_w: frame.width(),
        src_h: frame.height(),
        crop_x: box_x.max(0.0) as u32,
        crop_y: box_y.max(0.0) as u32,
        crop_w: box_w.max(1.0) as u32,
        crop_h: box_h.max(1.0) as u32,
        dest_w: dest,
        dest_h: dest,
        src_format: frame.format(),
        dest_format: ImageFormat::Rgbf32,
        fit_mode: FitMode::Stretch,
        rotation_degrees: 0.0,
        dest_layout: TensorLayout::Nchw,
    }
}

/// Mock CPU detector: one input `[1, 3, H, W]`, one output `[1, 4]` bounding box.
pub fn mock_cpu_detector(input_side: u32) -> MockCpuSession {
    let forward = Arc::new(move |_inputs: &[&Tensor<Cpu>]| {
        let shape = TensorShape::new([1, 4]).expect("valid shape");
        let boxes = vec![10.0f32, 20.0, 100.0, 120.0];
        cpu_tensor_f32(shape, &boxes).map(|t| vec![t])
    });
    MockCpuSession::new(
        vec![TensorShape::new([1, 3, input_side as usize, input_side as usize]).expect("valid shape")],
        vec![TensorShape::new([1, 4]).expect("valid shape")],
        forward,
    )
}

/// Mock CPU landmarker: one input `[1, 3, H, W]`, one output `[1, 4]` landmark pairs.
pub fn mock_cpu_landmarker(input_side: u32) -> MockCpuSession {
    let forward = Arc::new(move |_inputs: &[&Tensor<Cpu>]| {
        let shape = TensorShape::new([1, 4]).expect("valid shape");
        let points = vec![30.0f32, 40.0, 50.0, 60.0];
        cpu_tensor_f32(shape, &points).map(|t| vec![t])
    });
    MockCpuSession::new(
        vec![TensorShape::new([1, 3, input_side as usize, input_side as usize]).expect("valid shape")],
        vec![TensorShape::new([1, 4]).expect("valid shape")],
        forward,
    )
}

#[cfg(feature = "vulkan")]
pub fn mock_gpu_detector(device: Vulkan, input_side: u32) -> MockGpuSession {
    let forward = Arc::new(move |_inputs: &[&Tensor<Vulkan>]| {
        let shape = TensorShape::new([1, 4]).expect("valid shape");
        let boxes = vec![10.0f32, 20.0, 100.0, 120.0];
        cpu_tensor_f32(shape, &boxes).map(|t| vec![t])
    });
    MockGpuSession::new(
        device,
        vec![TensorShape::new([1, 3, input_side as usize, input_side as usize]).expect("valid shape")],
        vec![TensorShape::new([1, 4]).expect("valid shape")],
        forward,
    )
}

#[cfg(feature = "vulkan")]
pub fn mock_gpu_landmarker(device: Vulkan, input_side: u32) -> MockGpuSession {
    let forward = Arc::new(move |_inputs: &[&Tensor<Vulkan>]| {
        let shape = TensorShape::new([1, 4]).expect("valid shape");
        let points = vec![30.0f32, 40.0, 50.0, 60.0];
        cpu_tensor_f32(shape, &points).map(|t| vec![t])
    });
    MockGpuSession::new(
        device,
        vec![TensorShape::new([1, 3, input_side as usize, input_side as usize]).expect("valid shape")],
        vec![TensorShape::new([1, 4]).expect("valid shape")],
        forward,
    )
}
