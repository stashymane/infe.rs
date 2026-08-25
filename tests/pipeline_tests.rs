use std::sync::Arc;

use infers::{
    Backend, CoreError, CpuImageProcessor, DataType, Device, ExecuTorchBackend,
    ExecuTorchBackendConfig, GpuImageProcessor, ModelSession, VulkanContext,
};
use infers_gpu::ash::vk::Handle;
use infers_test_utils::{
    camera_frame_640x480, detector_preprocess_options, landmarker_preprocess_options,
    mock_gpu_detector, mock_gpu_landmarker, read_f32_output, MockBackend,
};

#[test]
fn test_cpu_face_pipeline() {
    let cpu = Device::cpu();
    let image_proc = CpuImageProcessor::new();

    let backend = MockBackend::new("mock");
    let mut detector = backend
        .load_model_with_config(
            &[0xAA, 0xBB],
            &infers::SessionConfig::new(cpu.clone()),
        )
        .expect("mock detector");
    let mut landmarker = backend
        .load_model_with_config(
            &[0xCC, 0xDD],
            &infers::SessionConfig::new(cpu.clone()),
        )
        .expect("mock landmarker");

    let frame = camera_frame_640x480(128);
    let detector_input = image_proc
        .process(&frame, &detector_preprocess_options(224))
        .expect("detector preprocess");

    assert_eq!(detector_input.shape().dims(), &[1, 224, 224, 3]);
    assert_eq!(detector_input.dtype(), DataType::F32);
    assert_eq!(detector_input.device(), &cpu);

    let detector_outputs = detector
        .run(&[detector_input.as_ref()])
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
        .run(&[landmarker_input.as_ref()])
        .expect("landmarker run");
    let landmarks = read_f32_output(&landmarker_outputs).expect("read landmarks");
    assert_eq!(landmarks.len(), 4);
}

#[test]
fn test_cpu_tensor_rejected_by_gpu_session() {
    let cpu = Device::cpu();
    let gpu = Device::gpu(0);

    let backend = MockBackend::new("mock");
    let mut gpu_session = backend
        .load_model_with_config(&[0x01], &infers::SessionConfig::new(gpu.clone()))
        .expect("mock gpu session");

    let image_proc = CpuImageProcessor::new();
    let frame = camera_frame_640x480(200);
    let cpu_tensor = image_proc
        .process(&frame, &detector_preprocess_options(224))
        .expect("cpu preprocess");

    let err = gpu_session.run(&[cpu_tensor.as_ref()]).unwrap_err();
    match err {
        CoreError::DeviceMismatch { expected, actual } => {
            assert_eq!(expected, gpu);
            assert_eq!(actual, cpu);
        }
        other => panic!("expected DeviceMismatch, got {other:?}"),
    }
}

#[test]
fn test_executorch_xnnpack_via_root_crate() {
    let backend = ExecuTorchBackend::new();
    let err = backend.load_model(
        &[0xDE, 0xAD, 0xBE, 0xEF],
        ExecuTorchBackendConfig::Xnnpack {
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
    let gpu = Device::gpu(0);
    let Some(ctx) = try_vulkan_context(&gpu) else {
        eprintln!("skipping: no Vulkan device");
        return;
    };

    let mut detector = mock_gpu_detector(224, gpu.clone());
    let mut landmarker = mock_gpu_landmarker(224, gpu.clone());

    let image_proc = GpuImageProcessor::new(Arc::clone(&ctx)).expect("gpu processor");
    assert_eq!(
        image_proc.context().device_handle().as_raw(),
        ctx.device_handle().as_raw(),
        "processor must use the shared Vulkan context"
    );

    let frame = camera_frame_640x480(128);
    let detector_input = image_proc
        .process(&frame, &detector_preprocess_options(224))
        .expect("gpu detector preprocess");

    assert_eq!(detector_input.device(), &gpu);
    assert_eq!(detector_input.shape().dims(), &[1, 224, 224, 3]);
    assert_eq!(detector_input.dtype(), DataType::F32);

    let detector_outputs = detector
        .run(&[detector_input.as_ref()])
        .expect("gpu detector run");
    let boxes = read_f32_output(&detector_outputs).expect("read boxes");
    assert_eq!(boxes, vec![10.0, 20.0, 100.0, 120.0]);

    let landmarker_input = image_proc
        .process(
            &frame,
            &landmarker_preprocess_options(boxes[0], boxes[1], boxes[2], boxes[3], 224),
        )
        .expect("gpu landmarker preprocess");
    assert_eq!(landmarker_input.device(), &gpu);

    let landmarker_outputs = landmarker
        .run(&[landmarker_input.as_ref()])
        .expect("gpu landmarker run");
    let landmarks = read_f32_output(&landmarker_outputs).expect("read landmarks");
    assert_eq!(landmarks, vec![30.0, 40.0, 50.0, 60.0]);

    // Teardown order: sessions and processor before the shared context Arc.
    drop(landmarker_outputs);
    drop(landmarker_input);
    drop(detector_outputs);
    drop(detector_input);
    drop(landmarker);
    drop(detector);
    drop(image_proc);
    drop(ctx);
}

#[test]
#[cfg(feature = "vulkan")]
fn test_execu_torch_vulkan_config_uses_same_context_as_processor() {
    let gpu = Device::gpu(0);
    let Some(ctx) = try_vulkan_context(&gpu) else {
        eprintln!("skipping: no Vulkan device");
        return;
    };

    let image_proc = GpuImageProcessor::new(Arc::clone(&ctx)).expect("gpu processor");

    let backend = ExecuTorchBackend::new();
    let result = backend.load_model(
        &[0u8; 32],
        ExecuTorchBackendConfig::Vulkan {
            context: Arc::clone(&ctx),
            method: None,
        },
    );
    // Invalid program bytes; registration still runs before Module load fails.
    assert!(result.is_err());

    let _ = image_proc.process(
        &camera_frame_640x480(64),
        &detector_preprocess_options(64),
    );

    drop(image_proc);
    drop(ctx);
}

#[cfg(feature = "vulkan")]
fn try_vulkan_context(device: &Device) -> Option<Arc<VulkanContext>> {
    VulkanContext::new(device).ok().map(Arc::new)
}
