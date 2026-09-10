//! End-to-end inference tests using real ExecuTorch models (no session mocks).

use infers::{
    CpuImageProcessor, DataType, ExecuTorchBackend, GpuImageProcessor, HardwareImageVulkanExt,
    Session, Vulkan, VulkanOptions, XnnpackOptions,
};
use infers_processing::DeferredCpuProcessExt;
#[cfg(feature = "vulkan")]
use infers_processing::DeferredVulkanProcessExt;
use infers_test_utils::{
    assets::{
        assert_detector_inference_outputs, read_model_bytes, yolo26n_face_asset, yolo26n_face_imgsz,
    },
    detector_preprocess_options, test_input_frames,
};

#[test]
fn test_xnnpack_detector_pipeline_inference() {
    let Some(pte_path) = yolo26n_face_asset("xnnpack/model.pte") else {
        eprintln!("skipping: target/yolo26n-face/xnnpack/model.pte not built");
        return;
    };
    let Some(imgsz) = yolo26n_face_imgsz() else {
        eprintln!("skipping: yolo26n-face manifest missing imgsz");
        return;
    };

    let bytes = read_model_bytes(&pte_path).expect("read xnnpack model.pte");
    let backend = ExecuTorchBackend::new();
    let mut session = backend
        .load_xnnpack(
            &bytes,
            XnnpackOptions {
                num_threads: 1,
                method: None,
            },
        )
        .expect("load xnnpack model");

    let image_proc = CpuImageProcessor::new();
    let expected_shapes = session.output_shapes().to_vec();

    for (label, frame) in test_input_frames() {
        let options = detector_preprocess_options(&frame, imgsz);
        let pending = frame
            .on_cpu()
            .process(&image_proc, &options)
            .unwrap_or_else(|e| panic!("{label}: cpu preprocess: {e}"));

        assert_eq!(
            pending.shape().dims(),
            &[1, 3, imgsz as usize, imgsz as usize],
            "{label}"
        );
        assert_eq!(pending.dtype(), DataType::F32, "{label}");

        let outputs = session
            .infer(pending)
            .unwrap_or_else(|e| panic!("{label}: xnnpack detector inference: {e}"));

        assert_detector_inference_outputs(&outputs, &expected_shapes);
    }
}

#[test]
#[cfg(feature = "vulkan")]
fn test_vulkan_detector_pipeline_inference() {
    let Some(pte_path) = yolo26n_face_asset("vulkan/model.pte") else {
        eprintln!("skipping: target/yolo26n-face/vulkan/model.pte not built");
        return;
    };
    let Some(imgsz) = yolo26n_face_imgsz() else {
        eprintln!("skipping: yolo26n-face manifest missing imgsz");
        return;
    };
    let vulkan = match Vulkan::new(0) {
        Ok(device) => device,
        Err(err) => {
            eprintln!("skipping: no Vulkan device ({err})");
            return;
        }
    };

    let bytes = read_model_bytes(&pte_path).expect("read vulkan model.pte");
    let backend = ExecuTorchBackend::new();
    let mut session = backend
        .load_vulkan(&bytes, &vulkan, VulkanOptions { method: None })
        .expect("load vulkan model");

    let image_proc = GpuImageProcessor::new(vulkan.clone()).expect("gpu processor");
    let expected_shapes = session.output_shapes().to_vec();

    for (label, frame) in test_input_frames() {
        let options = detector_preprocess_options(&frame, imgsz);
        let pending = frame
            .on(&vulkan)
            .process(&image_proc, &options)
            .unwrap_or_else(|e| panic!("{label}: gpu preprocess: {e}"));

        assert_eq!(
            pending.shape().dims(),
            &[1, 3, imgsz as usize, imgsz as usize],
            "{label}"
        );
        assert_eq!(pending.dtype(), DataType::F32, "{label}");

        let outputs = session
            .infer(pending)
            .unwrap_or_else(|e| panic!("{label}: vulkan detector inference: {e}"));

        assert_detector_inference_outputs(&outputs, &expected_shapes);
    }
}
