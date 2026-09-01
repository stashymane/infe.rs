use infers::{
    CoreError, CpuImageProcessor, DataType, Device, ExecuTorchBackend,
    Session, XnnpackOptions,
};
#[cfg(feature = "vulkan")]
use infers::{GpuImageProcessor, Vulkan};
use infers_test_utils::{
    camera_frame_640x480, detector_preprocess_options, landmarker_preprocess_options,
    mock_cpu_detector, mock_cpu_landmarker, mock_gpu_detector, mock_gpu_landmarker,
    read_f32_output,
};

#[test]
fn test_cpu_face_pipeline() {
    let image_proc = CpuImageProcessor::new();
    let mut detector = mock_cpu_detector(224);
    let mut landmarker = mock_cpu_landmarker(224);

    let frame = camera_frame_640x480(128);
    let detector_input = image_proc
        .process(&frame, &detector_preprocess_options(224))
        .expect("detector preprocess");

    assert_eq!(detector_input.shape().dims(), &[1, 3, 224, 224]);
    assert_eq!(detector_input.dtype(), DataType::F32);

    let detector_outputs = detector
        .run(&[&detector_input])
        .expect("detector run");
    let boxes = read_f32_output(&detector_outputs).expect("read boxes");
    assert_eq!(boxes.len(), 4);

    let landmarker_input = image_proc
        .process(
            &frame,
            &landmarker_preprocess_options(boxes[0], boxes[1], boxes[2], boxes[3], 224),
        )
        .expect("landmarker preprocess");

    let landmarker_outputs = landmarker
        .run(&[&landmarker_input])
        .expect("landmarker run");
    let landmarks = read_f32_output(&landmarker_outputs).expect("read landmarks");
    assert_eq!(landmarks.len(), 4);
}

#[test]
#[cfg(feature = "vulkan")]
fn test_cpu_tensor_rejected_by_gpu_session_is_compile_time() {
    // `MockGpuSession::run` accepts `&[&Tensor<Vulkan>]`; passing `Tensor<Cpu>` is a
    // compile error. See `crates/core/tests/ui/` trybuild suite.
}

#[test]
fn test_executorch_xnnpack_via_root_crate() {
    let backend = ExecuTorchBackend::new();
    let err = backend.load_xnnpack(
        &[0xDE, 0xAD, 0xBE, 0xEF],
        XnnpackOptions {
            num_threads: 1,
            method: None,
        },
    );
    assert!(err.is_err());
    match err.err().unwrap() {
        CoreError::ModelLoadFailed(_) => {}
        other => panic!("expected ModelLoadFailed, got {other:?}"),
    }
}

#[test]
#[cfg(feature = "vulkan")]
fn test_gpu_shared_vulkan_face_pipeline() {
    let vulkan = match Vulkan::new(0) {
        Ok(v) => v,
        Err(_) => {
            eprintln!("skipping: no Vulkan device");
            return;
        }
    };

    let mut detector = mock_gpu_detector(vulkan.clone(), 224);
    let mut landmarker = mock_gpu_landmarker(vulkan.clone(), 224);

    let image_proc = GpuImageProcessor::new(vulkan.clone()).expect("gpu processor");

    let frame = camera_frame_640x480(128);
    let gpu_image = vulkan.upload_image(&frame).expect("upload frame");
    let detector_input = image_proc
        .process(&gpu_image, &detector_preprocess_options(224))
        .expect("gpu detector preprocess");

    assert_eq!(detector_input.shape().dims(), &[1, 3, 224, 224]);
    assert_eq!(detector_input.dtype(), DataType::F32);

    let detector_outputs = detector
        .run(&[&detector_input])
        .expect("gpu detector run");
    let boxes = read_f32_output(&detector_outputs).expect("read boxes");
    assert_eq!(boxes, vec![10.0, 20.0, 100.0, 120.0]);

    let landmarker_input = image_proc
        .process(
            &vulkan.upload_image(&frame).expect("upload"),
            &landmarker_preprocess_options(boxes[0], boxes[1], boxes[2], boxes[3], 224),
        )
        .expect("gpu landmarker preprocess");

    let landmarker_outputs = landmarker
        .run(&[&landmarker_input])
        .expect("gpu landmarker run");
    let landmarks = read_f32_output(&landmarker_outputs).expect("read landmarks");
    assert_eq!(landmarks, vec![30.0, 40.0, 50.0, 60.0]);
}

#[test]
#[cfg(feature = "vulkan")]
fn test_execu_torch_vulkan_load_shares_context_with_processor() {
    let vulkan = match Vulkan::new(0) {
        Ok(v) => v,
        Err(_) => {
            eprintln!("skipping: no Vulkan device");
            return;
        }
    };

    let image_proc = GpuImageProcessor::new(vulkan.clone()).expect("gpu processor");

    let backend = ExecuTorchBackend::new();
    let result = backend.load_vulkan(
        &[0u8; 32],
        &vulkan,
        infers::VulkanOptions { method: None },
    );
    assert!(result.is_err());

    let gpu_image = vulkan
        .upload_image(&camera_frame_640x480(64))
        .expect("upload");
    let _ = image_proc
        .process(&gpu_image, &detector_preprocess_options(64))
        .expect("preprocess");

    drop(image_proc);
}
