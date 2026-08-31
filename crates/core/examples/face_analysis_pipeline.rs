use std::sync::Arc;
use infers_core::{
    Backend, CoreError, CpuImageBuffer, Device, FitMode, ImageFormat, ImageInputBuffer,
    ModelSession, ProcessingOptions, Rotation, TensorBuffer, TensorShape,
};
use infers_test_utils::{MockBackend, MockDeviceTensor, MockSession};

#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Clone)]
pub struct Landmark {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone)]
pub struct FaceAnalysisResult {
    pub bounding_boxes: Vec<BoundingBox>,
    pub landmarks: Vec<Vec<Landmark>>,
}

/// Simulated mock GPU image processor for the example pipeline
pub struct MockGpuImageProcessor {
    device: Device,
}

impl MockGpuImageProcessor {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
        }
    }

    pub fn process(
        &self,
        _input: &dyn ImageInputBuffer,
        options: &ProcessingOptions,
    ) -> Result<Box<dyn TensorBuffer>, CoreError> {
        // Output tensor shaped [1, 3, dest_h, dest_w]
        let shape = TensorShape::new([1, 3, options.dest_h as usize, options.dest_w as usize])
            .expect("valid shape");
        let element_count = shape.element_count();
        let dummy_data = vec![0.5f32; element_count];
        let tensor = MockDeviceTensor::from_f32_slice(self.device.clone(), shape, &dummy_data);
        Ok(Box::new(tensor))
    }
}

/// End-to-end Face Analysis Pipeline
pub struct FaceAnalysisPipeline {
    detector_session: Box<dyn ModelSession>,
    landmarker_session: Box<dyn ModelSession>,
    image_processor: MockGpuImageProcessor,
}

impl FaceAnalysisPipeline {
    pub fn new(_backend: &dyn Backend, device: &Device) -> Result<Self, CoreError> {
        // Build mock detector session
        let detector_dev = device.clone();
        let detector_forward = Arc::new(move |_inputs: &[&dyn TensorBuffer]| {
            // Detector outputs 1 face bounding box: [x, y, w, h]
            let shape = TensorShape::new([1, 4]).expect("valid shape");
            let boxes = vec![50.0f32, 50.0, 100.0, 100.0];
            let tensor = MockDeviceTensor::from_f32_slice(detector_dev.clone(), shape, &boxes);
            Ok(vec![Box::new(tensor) as Box<dyn TensorBuffer>])
        });

        let detector_session = Box::new(MockSession::new(
            device.clone(),
            vec![TensorShape::new([1, 3, 256, 256]).expect("valid shape")],
            vec![TensorShape::new([1, 4]).expect("valid shape")],
            detector_forward,
        ));

        // Build mock landmarker session
        let landmarker_dev = device.clone();
        let landmarker_forward = Arc::new(move |_inputs: &[&dyn TensorBuffer]| {
            // Landmarker outputs 5 facial points (10 f32 values)
            let shape = TensorShape::new([1, 10]).expect("valid shape");
            let points = vec![
                60.0f32, 60.0, // Left eye
                80.0, 60.0,    // Right eye
                70.0, 75.0,    // Nose
                65.0, 90.0,    // Mouth left
                75.0, 90.0,    // Mouth right
            ];
            let tensor = MockDeviceTensor::from_f32_slice(landmarker_dev.clone(), shape, &points);
            Ok(vec![Box::new(tensor) as Box<dyn TensorBuffer>])
        });

        let landmarker_session = Box::new(MockSession::new(
            device.clone(),
            vec![TensorShape::new([1, 3, 192, 192]).expect("valid shape")],
            vec![TensorShape::new([1, 10]).expect("valid shape")],
            landmarker_forward,
        ));

        let image_processor = MockGpuImageProcessor::new(device);

        Ok(Self {
            detector_session,
            landmarker_session,
            image_processor,
        })
    }

    pub fn process_frame(
        &mut self,
        input_image: &dyn ImageInputBuffer,
    ) -> Result<FaceAnalysisResult, CoreError> {
        println!("1. Processing camera frame on GPU for Face Detector (256x256 RGBF32)...");
        let detector_input = self.image_processor.process(
            input_image,
            &ProcessingOptions {
                src_w: input_image.width(),
                src_h: input_image.height(),
                crop_x: 0,
                crop_y: 0,
                crop_w: 0,
                crop_h: 0,
                dest_w: 256,
                dest_h: 256,
                src_format: ImageFormat::Rgb888,
                dest_format: ImageFormat::Rgbf32,
                fit_mode: FitMode::Contain,
                rotation: Rotation::None,
                dest_layout: infers_core::TensorLayout::Nchw,
            },
        )?;

        println!("2. Running Face Detector forward inference on GPU...");
        let detector_outputs = self.detector_session.run(&[detector_input.as_ref()])?;

        println!("3. Explicit CPU read for bounding boxes...");
        let boxes_host = detector_outputs[0].read_to_cpu()?;
        let boxes_slice = boxes_host.as_slice_f32()?;
        let detected_boxes = vec![BoundingBox {
            x: boxes_slice[0],
            y: boxes_slice[1],
            width: boxes_slice[2],
            height: boxes_slice[3],
        }];

        let mut all_landmarks = Vec::new();

        println!("4. Processing face crops on GPU for Landmarker (192x192 RGBF32)...");
        for bbox in &detected_boxes {
            let landmarker_input = self.image_processor.process(
                input_image,
                &ProcessingOptions {
                    src_w: input_image.width(),
                    src_h: input_image.height(),
                    crop_x: bbox.x as u32,
                    crop_y: bbox.y as u32,
                    crop_w: bbox.width as u32,
                    crop_h: bbox.height as u32,
                    dest_w: 192,
                    dest_h: 192,
                    src_format: ImageFormat::Rgb888,
                    dest_format: ImageFormat::Rgbf32,
                    fit_mode: FitMode::Stretch,
                    rotation: Rotation::None,
                    dest_layout: infers_core::TensorLayout::Nchw,
                },
            )?;

            println!("5. Running Landmarker forward inference on GPU...");
            let landmarker_outputs = self.landmarker_session.run(&[landmarker_input.as_ref()])?;

            println!("6. Explicit CPU read of landmarks...");
            let landmarks_host = landmarker_outputs[0].read_to_cpu()?;
            let landmarks_slice = landmarks_host.as_slice_f32()?;

            let mut face_landmarks = Vec::new();
            for chunk in landmarks_slice.chunks_exact(2) {
                face_landmarks.push(Landmark {
                    x: chunk[0],
                    y: chunk[1],
                });
            }
            all_landmarks.push(face_landmarks);
        }

        Ok(FaceAnalysisResult {
            bounding_boxes: detected_boxes,
            landmarks: all_landmarks,
        })
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Initializing Pipeline on GPU:0...");
    let gpu = Device::gpu(0);
    let backend = MockBackend::new("mock-executorch");

    let mut pipeline = FaceAnalysisPipeline::new(&backend, &gpu)?;

    // Create a 1920x1080 input image buffer
    let raw_image_bytes = vec![128u8; 1920 * 1080 * 3];
    let input_frame = CpuImageBuffer::new(1920, 1080, ImageFormat::Rgb888, raw_image_bytes)?;

    println!("Running frame through FaceAnalysisPipeline...");
    let result = pipeline.process_frame(&input_frame)?;

    println!("Pipeline execution succeeded!");
    println!("Detected faces: {:?}", result.bounding_boxes);
    println!("Detected landmarks: {:?}", result.landmarks);

    Ok(())
}
