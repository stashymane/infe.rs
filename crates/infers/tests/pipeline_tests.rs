use infers::{
    CoreError, CpuImageProcessor, DataType, ExecuTorchBackend, Session, XnnpackOptions,
};
use infers_processing::DeferredCpuProcessExt;
#[cfg(feature = "vulkan")]
use infers::{GpuImageProcessor, HardwareImageVulkanExt, Vulkan};
#[cfg(feature = "vulkan")]
use infers_processing::DeferredVulkanProcessExt;
use infers_test_utils::{
    detector_preprocess_options, landmarker_preprocess_options, mock_cpu_detector,
    mock_cpu_landmarker, mock_gpu_detector, mock_gpu_landmarker, read_f32_output, test_input_frames,
};

#[test]
fn test_cpu_face_pipeline() {
    let image_proc = CpuImageProcessor::new();

    for (label, frame) in test_input_frames() {
        let mut detector = mock_cpu_detector(224);
        let mut landmarker = mock_cpu_landmarker(224);

        let detector_pending = frame
            .clone()
            .on_cpu()
            .process(&image_proc, &detector_preprocess_options(&frame, 224))
            .unwrap_or_else(|e| panic!("{label}: detector preprocess: {e}"));

        assert_eq!(
            detector_pending.shape().dims(),
            &[1, 3, 224, 224],
            "{label}"
        );
        assert_eq!(detector_pending.dtype(), DataType::F32, "{label}");

        let detector_outputs = detector
            .infer(detector_pending)
            .unwrap_or_else(|e| panic!("{label}: detector infer: {e}"));
        let boxes = read_f32_output(&detector_outputs)
            .unwrap_or_else(|e| panic!("{label}: read boxes: {e}"));
        assert_eq!(boxes.len(), 4, "{label}");

        let landmarker_pending = frame
            .clone()
            .on_cpu()
            .process(
                &image_proc,
                &landmarker_preprocess_options(
                    &frame,
                    boxes[0],
                    boxes[1],
                    boxes[2],
                    boxes[3],
                    224,
                ),
            )
            .unwrap_or_else(|e| panic!("{label}: landmarker preprocess: {e}"));

        let landmarker_outputs = landmarker
            .infer(landmarker_pending)
            .unwrap_or_else(|e| panic!("{label}: landmarker infer: {e}"));
        let landmarks = read_f32_output(&landmarker_outputs)
            .unwrap_or_else(|e| panic!("{label}: read landmarks: {e}"));
        assert_eq!(landmarks.len(), 4, "{label}");
    }
}

#[test]
#[cfg(feature = "vulkan")]
fn test_cpu_tensor_rejected_by_gpu_session_is_compile_time() {
    // `MockGpuSession::run` accepts `&[&Tensor<Vulkan>]`; passing `Tensor<Cpu>` is a
    // compile error. See `crates/infers-core/tests/ui/` trybuild suite.
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

    let image_proc = GpuImageProcessor::new(vulkan.clone()).expect("gpu processor");

    for (label, frame) in test_input_frames() {
        let mut detector = mock_gpu_detector(vulkan.clone(), 224);
        let mut landmarker = mock_gpu_landmarker(vulkan.clone(), 224);

        let detector_pending = frame
            .clone()
            .on(&vulkan)
            .process(&image_proc, &detector_preprocess_options(&frame, 224))
            .unwrap_or_else(|e| panic!("{label}: gpu detector preprocess: {e}"));

        assert_eq!(
            detector_pending.shape().dims(),
            &[1, 3, 224, 224],
            "{label}"
        );
        assert_eq!(detector_pending.dtype(), DataType::F32, "{label}");

        let detector_outputs = detector
            .infer(detector_pending)
            .unwrap_or_else(|e| panic!("{label}: gpu detector infer: {e}"));
        let boxes = read_f32_output(&detector_outputs)
            .unwrap_or_else(|e| panic!("{label}: read boxes: {e}"));
        assert_eq!(boxes, vec![10.0, 20.0, 100.0, 120.0], "{label}");

        let landmarker_pending = frame
            .clone()
            .on(&vulkan)
            .process(
                &image_proc,
                &landmarker_preprocess_options(
                    &frame,
                    boxes[0],
                    boxes[1],
                    boxes[2],
                    boxes[3],
                    224,
                ),
            )
            .unwrap_or_else(|e| panic!("{label}: gpu landmarker preprocess: {e}"));

        let landmarker_outputs = landmarker
            .infer(landmarker_pending)
            .unwrap_or_else(|e| panic!("{label}: gpu landmarker infer: {e}"));
        let landmarks = read_f32_output(&landmarker_outputs)
            .unwrap_or_else(|e| panic!("{label}: read landmarks: {e}"));
        assert_eq!(landmarks, vec![30.0, 40.0, 50.0, 60.0], "{label}");
    }
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

    for (label, frame) in test_input_frames() {
        let options = detector_preprocess_options(&frame, 64);
        let _ = frame
            .on(&vulkan)
            .process(&image_proc, &options)
            .unwrap_or_else(|e| panic!("{label}: preprocess: {e}"));
    }

    drop(image_proc);
}
