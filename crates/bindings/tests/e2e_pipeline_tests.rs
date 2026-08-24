use infers_bindings::*;
use std::time::Instant;

mod common;

#[test]
fn test_end_to_end_face_pipeline_uniffi() {
    let cpu = create_cpu_device();

    // 1. Initialize image processor
    let image_proc = create_cpu_image_processor();

    // 2. Mock detector session on CPU
    let detector = common::load_mock_session(vec![0xAA, 0xBB], cpu.clone());
    let landmarker = common::load_mock_session(vec![0xCC, 0xDD], cpu.clone());

    // 3. Simulate camera stream frame (640x480 RGB888)
    let frame_data = vec![128u8; 640 * 480 * 3];
    let hw_buffer = common::create_mock_hardware_buffer(640, 480, ImageFormat::Rgb888, frame_data, cpu.clone());

    // 4. Step 1: Preprocess camera frame -> Face detector input (224x224 RGBF32)
    let detector_options = ProcessingOptions {
        src_w: 640,
        src_h: 480,
        crop_x: 0,
        crop_y: 0,
        crop_w: 640,
        crop_h: 480,
        dest_w: 224,
        dest_h: 224,
        dest_format: ImageFormat::Rgbf32,
        fit_mode: FitMode::Contain,
        rotation: Rotation::None,
    };

    let detector_input = image_proc
        .process_hardware_buffer(hw_buffer.clone(), detector_options)
        .expect("Preprocessing for detector failed");

    assert_eq!(
        detector_input.shape(),
        TensorShape {
            dims: vec![1, 224, 224, 3]
        }
    );
    assert_eq!(detector_input.dtype(), DataType::F32);

    // 5. Run Detector
    let detector_outputs = detector
        .run(vec![detector_input])
        .expect("Detector execution failed");
    assert_eq!(detector_outputs.len(), 1);

    // 6. Explicit CPU read for bounding boxes
    let raw_boxes = detector_outputs[0]
        .read_to_cpu_f32()
        .expect("Failed to read bounding boxes to CPU");
    assert_eq!(raw_boxes.len(), 4);
    let (box_x, box_y, box_w, box_h) = (raw_boxes[0], raw_boxes[1], raw_boxes[2], raw_boxes[3]);

    // 7. Step 2: Crop detected face from original camera frame -> Landmarker input (128x128 RGBF32)
    let landmarker_options = ProcessingOptions {
        src_w: 640,
        src_h: 480,
        crop_x: box_x.max(0.0) as u32,
        crop_y: box_y.max(0.0) as u32,
        crop_w: box_w.max(1.0) as u32,
        crop_h: box_h.max(1.0) as u32,
        dest_w: 224,
        dest_h: 224,
        dest_format: ImageFormat::Rgbf32,
        fit_mode: FitMode::Stretch,
        rotation: Rotation::None,
    };

    let landmarker_input = image_proc
        .process_hardware_buffer(hw_buffer, landmarker_options)
        .expect("Preprocessing for landmarker failed");

    // 8. Run Landmarker
    let landmarker_outputs = landmarker
        .run(vec![landmarker_input])
        .expect("Landmarker execution failed");
    assert_eq!(landmarker_outputs.len(), 1);

    // 9. Explicit CPU read for landmarks
    let raw_landmarks = landmarker_outputs[0]
        .read_to_cpu_f32()
        .expect("Failed to read landmarks to CPU");
    assert_eq!(raw_landmarks.len(), 4);
}

#[test]
fn test_cross_device_mismatch_error_handling() {
    let cpu = create_cpu_device();
    let gpu = create_gpu_device(0);

    let gpu_session = common::load_mock_session(vec![0x01, 0x02], gpu.clone());

    // Create CPU tensor
    let cpu_tensor = create_tensor_from_f32(
        TensorShape {
            dims: vec![1, 3, 224, 224],
        },
        vec![0.0f32; 1 * 3 * 224 * 224],
    )
    .expect("Failed to create CPU tensor");

    assert_eq!(cpu_tensor.device(), cpu);

    // Passing CPU tensor into GPU session must fail with DeviceMismatch error
    let result = gpu_session.run(vec![cpu_tensor]);
    assert!(result.is_err());
    match result.unwrap_err() {
        InfersError::DeviceMismatch { expected, actual } => {
            assert_eq!(expected, "GPU:0");
            assert_eq!(actual, "CPU");
        }
        other => panic!("Expected DeviceMismatch error, got {:?}", other),
    }
}

#[test]
fn test_pipeline_latency_and_throughput_benchmark() {
    let cpu = create_cpu_device();
    let image_proc = create_cpu_image_processor();
    let detector = common::load_mock_session(vec![0x10, 0x20], cpu.clone());

    let frame_data = vec![200u8; 320 * 240 * 3];
    let hw_buffer = common::create_mock_hardware_buffer(320, 240, ImageFormat::Rgb888, frame_data, cpu);

    let options = ProcessingOptions {
        src_w: 320,
        src_h: 240,
        crop_x: 0,
        crop_y: 0,
        crop_w: 320,
        crop_h: 240,
        dest_w: 224,
        dest_h: 224,
        dest_format: ImageFormat::Rgbf32,
        fit_mode: FitMode::Contain,
        rotation: Rotation::None,
    };

    let iterations = 20;
    let start = Instant::now();

    for _ in 0..iterations {
        let input_tensor = image_proc
            .process_hardware_buffer(hw_buffer.clone(), options.clone())
            .expect("Preprocess failed");
        let outputs = detector
            .run(vec![input_tensor])
            .expect("Detector run failed");
        let _host_boxes = outputs[0].read_to_cpu_f32().expect("Read to host failed");
    }

    let elapsed = start.elapsed();
    let avg_latency = elapsed / iterations;
    println!(
        "Pipeline benchmark: {} frames processed in {:?}, avg latency per frame: {:?}",
        iterations, elapsed, avg_latency
    );
    assert!(avg_latency.as_millis() < 50, "Average latency too high: {:?}", avg_latency);
}
