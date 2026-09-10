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
    camera_frame_640x480, detector_preprocess_options,
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
    let frame = camera_frame_640x480(128);
    let pending = frame
        .on_cpu()
        .process(&image_proc, &detector_preprocess_options(imgsz))
        .expect("cpu preprocess");

    assert_eq!(pending.shape().dims(), &[1, 3, imgsz as usize, imgsz as usize]);
    assert_eq!(pending.dtype(), DataType::F32);

    let outputs = session
        .infer(pending)
        .expect("xnnpack detector inference must complete");

    let expected_shapes = session.output_shapes().to_vec();
    assert_detector_inference_outputs(&outputs, &expected_shapes);
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
        .load_vulkan(
            &bytes,
            &vulkan,
            VulkanOptions { method: None },
        )
        .expect("load vulkan model");

    let image_proc = GpuImageProcessor::new(vulkan.clone()).expect("gpu processor");
    let frame = camera_frame_640x480(128);
    let pending = frame
        .on(&vulkan)
        .process(&image_proc, &detector_preprocess_options(imgsz))
        .expect("gpu preprocess");

    assert_eq!(pending.shape().dims(), &[1, 3, imgsz as usize, imgsz as usize]);
    assert_eq!(pending.dtype(), DataType::F32);

    let outputs = session
        .infer(pending)
        .expect("vulkan detector inference must complete");

    let expected_shapes = session.output_shapes().to_vec();
    assert_detector_inference_outputs(&outputs, &expected_shapes);
}
