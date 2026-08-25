use std::sync::Arc;

use infers::{
    CoreError, CpuImageBuffer, Device, FitMode, ImageFormat, ProcessingOptions, Rotation,
    TensorBuffer, TensorShape,
};
use infers_test_utils::{MockDeviceTensor, MockSession};

/// Solid-color 640×480 RGB camera frame for pipeline tests.
pub fn camera_frame_640x480(fill: u8) -> CpuImageBuffer {
    let bytes = vec![fill; 640 * 480 * 3];
    CpuImageBuffer::new(640, 480, ImageFormat::RGB888, bytes).expect("valid frame buffer")
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
        src_format: ImageFormat::RGB888,
        dest_format: ImageFormat::RGBF32,
        fit_mode: FitMode::CONTAIN,
        rotation: Rotation::None,
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
        src_format: ImageFormat::RGB888,
        dest_format: ImageFormat::RGBF32,
        fit_mode: FitMode::STRETCH,
        rotation: Rotation::None,
    }
}

/// Mock GPU detector: one input `[1, H, W, 3]`, one output `[1, 4]` bounding box.
pub fn mock_gpu_detector(input_side: u32, device: Device) -> MockSession {
    let session_device = device.clone();
    let forward = Arc::new(move |_inputs: &[&dyn TensorBuffer]| {
        let shape = TensorShape::from([1, 4]);
        let boxes = vec![10.0f32, 20.0, 100.0, 120.0];
        Ok(vec![Box::new(MockDeviceTensor::from_f32_slice(
            session_device.clone(),
            shape,
            &boxes,
        )) as Box<dyn TensorBuffer>])
    });
    MockSession::new(
        device,
        vec![TensorShape::from([1, input_side as usize, input_side as usize, 3])],
        vec![TensorShape::from([1, 4])],
        forward,
    )
}

/// Mock GPU landmarker: one input `[1, H, W, 3]`, one output `[1, 4]` landmark pairs.
pub fn mock_gpu_landmarker(input_side: u32, device: Device) -> MockSession {
    let session_device = device.clone();
    let forward = Arc::new(move |_inputs: &[&dyn TensorBuffer]| {
        let shape = TensorShape::from([1, 4]);
        let points = vec![30.0f32, 40.0, 50.0, 60.0];
        Ok(vec![Box::new(MockDeviceTensor::from_f32_slice(
            session_device.clone(),
            shape,
            &points,
        )) as Box<dyn TensorBuffer>])
    });
    MockSession::new(
        device,
        vec![TensorShape::from([1, input_side as usize, input_side as usize, 3])],
        vec![TensorShape::from([1, 4])],
        forward,
    )
}

pub fn read_f32_output(outputs: &[Box<dyn TensorBuffer>]) -> Result<Vec<f32>, CoreError> {
    let host = outputs[0].read_to_cpu()?;
    Ok(host.as_slice_f32()?.to_vec())
}
