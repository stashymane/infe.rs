use infers_backend_executorch::{
    data_type_to_scalar_type, scalar_type_to_data_type, ExecuTorchBackend, ExecuTorchError,
    ProgramMetadata, ScalarType, XnnpackOptions, VulkanOptions,
};
use infers_core::{CoreError, Cpu, DataType, HostTensor, Session, Tensor, TensorShape};
use infers_test_utils::assets::{
    assert_detector_inference_outputs, detector_input_zeros, read_model_bytes, yolo26n_face_asset,
    yolo26n_face_imgsz,
};
use std::sync::{Arc, OnceLock};

#[cfg(feature = "vulkan")]
use infers_gpu::Vulkan;

#[cfg(feature = "vulkan")]
fn shared_vulkan() -> Option<Vulkan> {
    static VULKAN: OnceLock<Option<Vulkan>> = OnceLock::new();
    VULKAN.get_or_init(|| Vulkan::new(0).ok()).clone()
}

fn manifest_imgsz(manifest: &std::path::Path) -> Option<u32> {
    let content = std::fs::read_to_string(manifest).ok()?;
    for line in content.lines() {
        let line = line.trim();
        if line.starts_with("imgsz:") {
            return line.split(':').nth(1)?.trim().parse().ok();
        }
    }
    None
}

fn assert_detection_input_shape(meta: &ProgramMetadata, imgsz: u32) {
    let forward = meta
        .methods
        .get("forward")
        .expect("forward method metadata");
    assert_eq!(forward.inputs.len(), 1);
    assert_eq!(forward.inputs[0].dtype, DataType::F32);
    assert_eq!(
        forward.inputs[0].shape.dims(),
        &[1, 3, imgsz as usize, imgsz as usize]
    );
    assert!(!forward.outputs.is_empty());
}

#[test]
fn test_data_type_scalar_type_conversions() {
    assert_eq!(
        data_type_to_scalar_type(DataType::F32).unwrap(),
        ScalarType::Float
    );
    assert_eq!(
        data_type_to_scalar_type(DataType::U8).unwrap(),
        ScalarType::Byte
    );
    assert_eq!(
        data_type_to_scalar_type(DataType::I32).unwrap(),
        ScalarType::Int
    );
    assert_eq!(
        data_type_to_scalar_type(DataType::I64).unwrap(),
        ScalarType::Long
    );
    assert_eq!(
        data_type_to_scalar_type(DataType::F16).unwrap(),
        ScalarType::Half
    );

    assert_eq!(
        scalar_type_to_data_type(ScalarType::Float).unwrap(),
        DataType::F32
    );
    assert_eq!(
        scalar_type_to_data_type(ScalarType::Byte).unwrap(),
        DataType::U8
    );
    assert_eq!(
        scalar_type_to_data_type(ScalarType::Int).unwrap(),
        DataType::I32
    );
    assert_eq!(
        scalar_type_to_data_type(ScalarType::Long).unwrap(),
        DataType::I64
    );
    assert_eq!(
        scalar_type_to_data_type(ScalarType::Half).unwrap(),
        DataType::F16
    );
}

#[test]
fn test_invalid_program_bytes_rejected_before_native_load() {
    let too_small = ProgramMetadata::from_bytes(&[0u8; 16]);
    assert!(too_small.is_err());
    match too_small.unwrap_err() {
        ExecuTorchError::InvalidProgram(msg) => {
            assert!(msg.contains("too small"));
        }
        other => panic!("expected InvalidProgram, got {other:?}"),
    }

    let wrong_magic = ProgramMetadata::from_bytes(&[0u8; 64]);
    assert!(wrong_magic.is_err());
    match wrong_magic.unwrap_err() {
        ExecuTorchError::InvalidProgram(msg) => {
            assert!(msg.contains("ET12"));
        }
        other => panic!("expected InvalidProgram, got {other:?}"),
    }
}

#[test]
fn test_cpu_tensor_from_host_and_readback() {
    let shape = TensorShape::new(vec![1, 3, 2, 2]).unwrap();
    let data: Vec<f32> = (0..12).map(|v| v as f32).collect();
    let host = HostTensor::from_f32(shape.clone(), data.clone()).unwrap();
    let tensor = Tensor::from_host(&Cpu, &host).unwrap();

    assert_eq!(tensor.device(), &Cpu);
    assert_eq!(tensor.shape(), &shape);
    assert_eq!(tensor.dtype(), DataType::F32);

    let readback = tensor.read_to_host().unwrap();
    assert_eq!(readback.as_slice_f32().unwrap(), &data[..]);
}

#[test]
fn test_load_invalid_model_xnnpack() {
    let backend = ExecuTorchBackend::new();
    let err = backend.load_xnnpack(
        &[0xDE, 0xAD, 0xBE, 0xEF],
        XnnpackOptions {
            num_threads: 1,
            method: None,
        },
    );
    assert!(err.is_err());
    let err = err.err().unwrap();
    assert!(
        matches!(err, CoreError::ModelLoadFailed(_)),
        "expected ModelLoadFailed, got {err:?}"
    );
}

#[test]
fn test_asset_model_loading_and_error_handling() {
    let backend = ExecuTorchBackend::new();
    let tflite_path = "../../assets/face_detector.tflite";

    let result = backend.load_xnnpack_from_file(
        tflite_path,
        XnnpackOptions {
            num_threads: 1,
            method: None,
        },
    );
    assert!(result.is_err());
    match result.err().unwrap() {
        CoreError::ModelLoadFailed(msg) => {
            assert!(!msg.is_empty());
        }
        other => panic!("Expected ModelLoadFailed error, got {:?}", other),
    }
}

#[test]
fn test_yolo26n_face_xnnpack_asset_loads() {
    let Some(pte_path) = yolo26n_face_asset("xnnpack/model.pte") else {
        eprintln!("skipping: target/yolo26n-face/xnnpack/model.pte not built");
        return;
    };
    let Some(manifest_path) = yolo26n_face_asset("manifest.yaml") else {
        eprintln!("skipping: target/yolo26n-face/manifest.yaml missing");
        return;
    };
    let Some(imgsz) = manifest_imgsz(&manifest_path) else {
        panic!("manifest.yaml missing imgsz");
    };

    let bytes = std::fs::read(&pte_path).expect("read xnnpack model.pte");
    let meta = ProgramMetadata::from_bytes(&bytes).expect("parse program metadata");
    assert_detection_input_shape(&meta, imgsz);

    let backend = ExecuTorchBackend::new();
    let session = backend
        .load_xnnpack(
            &bytes,
            XnnpackOptions {
                num_threads: 1,
                method: None,
            },
        )
        .expect("load xnnpack yolo26n-face asset");
    assert_eq!(session.input_shapes().len(), 1);
    assert_eq!(
        session.input_shapes()[0].dims(),
        &[1, 3, imgsz as usize, imgsz as usize]
    );
}

#[test]
#[cfg(feature = "vulkan")]
fn test_yolo26n_face_vulkan_asset_loads() {
    let Some(pte_path) = yolo26n_face_asset("vulkan/model.pte") else {
        eprintln!("skipping: target/yolo26n-face/vulkan/model.pte not built");
        return;
    };
    let Some(manifest_path) = yolo26n_face_asset("manifest.yaml") else {
        eprintln!("skipping: target/yolo26n-face/manifest.yaml missing");
        return;
    };
    let Some(imgsz) = manifest_imgsz(&manifest_path) else {
        panic!("manifest.yaml missing imgsz");
    };

    let Some(vulkan) = shared_vulkan() else {
        eprintln!("skipping: no Vulkan device");
        return;
    };

    let bytes = std::fs::read(&pte_path).expect("read vulkan model.pte");
    let meta = ProgramMetadata::from_bytes(&bytes).expect("parse vulkan program metadata");
    assert_detection_input_shape(&meta, imgsz);

    let backend = ExecuTorchBackend::new();
    let session = backend
        .load_vulkan(
            &bytes,
            &vulkan,
            VulkanOptions { method: None },
        )
        .expect("load vulkan yolo26n-face asset");
    assert_eq!(session.input_shapes().len(), 1);
    assert_eq!(
        session.input_shapes()[0].dims(),
        &[1, 3, imgsz as usize, imgsz as usize]
    );
}

#[test]
fn test_delegate_configuration_and_device_selection() {
    use infers_backend_executorch::ExecuTorchDelegate;

    let xnnpack = ExecuTorchDelegate::Xnnpack;
    assert_eq!(xnnpack.name(), "xnnpack");
    assert!(xnnpack.supports_device_kind(infers_core::DeviceKind::Cpu));
    assert!(!xnnpack.supports_device_kind(infers_core::DeviceKind::Gpu));

    let vulkan = ExecuTorchDelegate::Vulkan;
    assert_eq!(vulkan.name(), "vulkan");
    assert!(vulkan.supports_device_kind(infers_core::DeviceKind::Gpu));
    assert!(!vulkan.supports_device_kind(infers_core::DeviceKind::Cpu));

    let qnn = ExecuTorchDelegate::Qnn;
    assert_eq!(qnn.name(), "qnn");
    assert!(qnn.supports_device_kind(infers_core::DeviceKind::Npu));

    let coreml = ExecuTorchDelegate::CoreMl;
    assert_eq!(coreml.name(), "coreml");
    assert!(coreml.supports_device_kind(infers_core::DeviceKind::Gpu));
    assert!(coreml.supports_device_kind(infers_core::DeviceKind::Npu));
}

#[test]
#[cfg(feature = "vulkan")]
fn test_shared_vulkan_processor_uses_same_context() {
    use infers_gpu::ash::vk::Handle;
    use infers_processing::GpuImageProcessor;

    let Ok(vulkan) = Vulkan::new(0) else {
        eprintln!("skipping: no Vulkan device");
        return;
    };
    let ctx = Arc::clone(vulkan.context());
    let proc = GpuImageProcessor::new(vulkan.clone()).expect("processor");
    assert_eq!(
        proc.context().device_handle().as_raw(),
        ctx.device_handle().as_raw()
    );

    drop(proc);
}

#[test]
#[cfg(feature = "vulkan")]
fn test_prepare_gpu_inputs_accepts_vulkan_tensors() {
    use infers_backend_executorch::gpu_input;
    use infers_test_utils::gpu_tensor_f32;

    let Ok(vulkan) = Vulkan::new(0) else {
        eprintln!("skipping: no Vulkan device");
        return;
    };
    let shape = TensorShape::new(vec![1, 64, 64, 3]).unwrap();
    let gpu = gpu_tensor_f32(&vulkan, shape, &[0.0; 64 * 64 * 3]).expect("gpu tensor");

    let plan = gpu_input::prepare_inputs(&[&gpu], vulkan.context(), None).expect("prepare_inputs");
    assert_eq!(plan.evalues().len(), 1);
    assert_eq!(plan.skip_staging_mask, 0);
}

#[test]
#[cfg(feature = "vulkan")]
fn test_vulkan_config_registers_or_reports_missing_backend() {
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("skipping: no Vulkan device");
        return;
    };
    let backend = ExecuTorchBackend::new();
    let result = backend.load_vulkan(
        &[0u8; 16],
        &vulkan,
        VulkanOptions { method: None },
    );
    assert!(result.is_err());
}

#[test]
fn test_host_tensor_rejects_size_mismatch() {
    let shape = TensorShape::new([1, 4]).expect("valid shape");
    let err = HostTensor::new(shape, DataType::F32, vec![0u8; 8]).unwrap_err();
    match err {
        infers_core::CoreError::BufferTransferFailed(_)
        | infers_core::CoreError::InvalidArgument(_)
        | infers_core::CoreError::InvalidShape(_) => {}
        other => panic!("expected buffer error, got {other:?}"),
    }
}

#[test]
fn test_yolo26n_face_xnnpack_inference() {
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
        .expect("load xnnpack yolo26n-face asset");

    let input = detector_input_zeros(imgsz).expect("detector input tensor");
    let outputs = session
        .infer(&input)
        .expect("xnnpack inference must complete without crashing");

    let expected_shapes = session.output_shapes().to_vec();
    assert_detector_inference_outputs(&outputs, &expected_shapes);
}

#[test]
#[cfg(feature = "vulkan")]
fn test_yolo26n_face_vulkan_inference() {
    let Some(pte_path) = yolo26n_face_asset("vulkan/model.pte") else {
        eprintln!("skipping: target/yolo26n-face/vulkan/model.pte not built");
        return;
    };
    let Some(imgsz) = yolo26n_face_imgsz() else {
        eprintln!("skipping: yolo26n-face manifest missing imgsz");
        return;
    };
    let Some(vulkan) = shared_vulkan() else {
        eprintln!("skipping: no Vulkan device");
        return;
    };

    let bytes = read_model_bytes(&pte_path).expect("read vulkan model.pte");
    let backend = ExecuTorchBackend::new();
    let mut session = backend
        .load_vulkan(
            &bytes,
            &vulkan,
            VulkanOptions { method: None },
        )
        .expect("load vulkan yolo26n-face asset");

    let cpu_input = detector_input_zeros(imgsz).expect("detector input tensor");
    let gpu_input = cpu_input.to_device(&vulkan).expect("upload input to GPU");
    let outputs = session
        .infer(&gpu_input)
        .expect("vulkan inference must complete without crashing");

    let expected_shapes = session.output_shapes().to_vec();
    assert_detector_inference_outputs(&outputs, &expected_shapes);
}
