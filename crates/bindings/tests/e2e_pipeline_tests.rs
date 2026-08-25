use infers_bindings::*;
use std::time::Instant;

mod common;

fn bindings_processing_options(
    opts: infers_core::ProcessingOptions,
) -> ProcessingOptions {
    ProcessingOptions {
        src_w: opts.src_w,
        src_h: opts.src_h,
        crop_x: opts.crop_x,
        crop_y: opts.crop_y,
        crop_w: opts.crop_w,
        crop_h: opts.crop_h,
        dest_w: opts.dest_w,
        dest_h: opts.dest_h,
        src_format: opts.src_format.into(),
        dest_format: opts.dest_format.into(),
        fit_mode: opts.fit_mode.into(),
        rotation: opts.rotation.into(),
    }
}

#[test]
fn test_end_to_end_face_pipeline_uniffi() {
    let cpu = create_cpu_device();

    let image_proc = create_cpu_image_processor();

    let detector = common::load_mock_session(vec![0xAA, 0xBB], cpu.clone());
    let landmarker = common::load_mock_session(vec![0xCC, 0xDD], cpu.clone());

    let frame_data = vec![128u8; 640 * 480 * 3];

    let detector_options =
        bindings_processing_options(infers_test_utils::detector_preprocess_options(224));

    let detector_input = image_proc
        .process_bytes(frame_data.clone(), 640, 480, ImageFormat::Rgb888, detector_options)
        .expect("Preprocessing for detector failed");

    assert_eq!(
        detector_input.shape(),
        TensorShape {
            dims: vec![1, 224, 224, 3]
        }
    );
    assert_eq!(detector_input.dtype(), DataType::F32);

    let detector_outputs = detector
        .run(vec![detector_input])
        .expect("Detector execution failed");
    assert_eq!(detector_outputs.len(), 1);

    let raw_boxes = detector_outputs[0]
        .read_to_cpu_f32()
        .expect("Failed to read bounding boxes to CPU");
    assert_eq!(raw_boxes.len(), 4);
    let (box_x, box_y, box_w, box_h) = (raw_boxes[0], raw_boxes[1], raw_boxes[2], raw_boxes[3]);

    let landmarker_options = bindings_processing_options(
        infers_test_utils::landmarker_preprocess_options(box_x, box_y, box_w, box_h, 224),
    );

    let landmarker_input = image_proc
        .process_bytes(frame_data, 640, 480, ImageFormat::Rgb888, landmarker_options)
        .expect("Preprocessing for landmarker failed");

    let landmarker_outputs = landmarker
        .run(vec![landmarker_input])
        .expect("Landmarker execution failed");
    assert_eq!(landmarker_outputs.len(), 1);

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

    let cpu_tensor = create_tensor_from_f32(
        TensorShape {
            dims: vec![1, 3, 224, 224],
        },
        vec![0.0f32; 3 * 224 * 224],
    )
    .expect("Failed to create CPU tensor");

    assert_eq!(cpu_tensor.device(), cpu);

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

    let mut options = bindings_processing_options(
        infers_test_utils::detector_preprocess_options(224),
    );
    options.src_w = 320;
    options.src_h = 240;
    options.crop_w = 320;
    options.crop_h = 240;

    let iterations = 20;
    let start = Instant::now();

    for _ in 0..iterations {
        let input_tensor = image_proc
            .process_bytes(frame_data.clone(), 320, 240, ImageFormat::Rgb888, options.clone())
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
