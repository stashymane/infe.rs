use infers_backend_executorch::{
    data_type_to_scalar_type, scalar_type_to_data_type, ExecuTorchBackend,
    ExecuTorchBackendConfig, ExecuTorchError, ExecuTorchTensorBuffer, ProgramMetadata,
    ScalarType,
};
use infers_core::{Backend, CoreError, DataType, Device, ModelSession as _, TensorBuffer, TensorShape};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

#[cfg(feature = "vulkan")]
fn shared_vulkan_context() -> Option<Arc<infers_gpu::VulkanContext>> {
    static CTX: OnceLock<Option<Arc<infers_gpu::VulkanContext>>> = OnceLock::new();
    CTX.get_or_init(|| {
        infers_gpu::VulkanContext::new(&Device::gpu(0))
            .ok()
            .map(Arc::new)
    })
    .clone()
}


fn yolo26n_face_asset(subpath: &str) -> Option<PathBuf> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../assets/yolo26n-face")
        .join(subpath);
    path.exists().then_some(path)
}

fn manifest_imgsz(manifest: &Path) -> Option<u32> {
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
fn test_tensor_buffer_ndarray_and_host_readback() {
    let shape = TensorShape::new(vec![1, 3, 2, 2]).unwrap();
    let data: Vec<f32> = (0..12).map(|v| v as f32).collect();

    let tb = ExecuTorchTensorBuffer::from_f32_slice(Device::cpu(), shape.clone(), &data)
        .expect("Failed to create ExecuTorchTensorBuffer");

    assert_eq!(tb.device(), &Device::cpu());
    assert_eq!(tb.shape(), &shape);
    assert_eq!(tb.dtype(), DataType::F32);
    assert_eq!(tb.as_slice_f32().unwrap(), &data[..]);

    let ndarray = tb.to_ndarray_f32().expect("Failed to convert to ndarray");
    assert_eq!(ndarray.shape(), &[1, 3, 2, 2]);
    assert_eq!(ndarray[[0, 2, 1, 1]], 11.0);

    let host = tb.read_to_cpu().expect("Failed to read to CPU");
    let cpu_slice: &[f32] = host.as_slice_f32().expect("Failed to get f32 slice");
    assert_eq!(cpu_slice, &data[..]);
}

#[test]
fn test_executorch_backend_devices() {
    let backend = ExecuTorchBackend::new();
    let devices = backend.available_devices();
    assert_eq!(devices.len(), 3);
    assert_eq!(devices[0], Device::cpu());
    assert_eq!(devices[1], Device::gpu(0));
    assert_eq!(devices[2], Device::npu(0));
}

#[test]
fn test_load_invalid_model_xnnpack() {
    let backend = ExecuTorchBackend::new();
    let err = backend.load_model(
        &[0xDE, 0xAD, 0xBE, 0xEF],
        ExecuTorchBackendConfig::Xnnpack {
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
fn test_trait_gpu_load_requires_vulkan_config() {
    let backend = ExecuTorchBackend::new();
    let err = backend.load_model_with_config(
        &[0u8; 8],
        &infers_core::SessionConfig::new(Device::gpu(0)),
    );
    assert!(err.is_err());
    match err.err().unwrap() {
        CoreError::ModelLoadFailed(msg) => {
            assert!(msg.contains("Vulkan"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn test_asset_model_loading_and_error_handling() {
    let backend = ExecuTorchBackend::new();
    let tflite_path = "../../../assets/face_detector.tflite";

    let result = backend.load_model_from_file(
        tflite_path,
        ExecuTorchBackendConfig::Xnnpack {
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
        eprintln!("skipping: assets/yolo26n-face/xnnpack/model.pte not built");
        return;
    };
    let Some(manifest_path) = yolo26n_face_asset("manifest.yaml") else {
        eprintln!("skipping: assets/yolo26n-face/manifest.yaml missing");
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
        .load_model(
            &bytes,
            ExecuTorchBackendConfig::Xnnpack {
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
        eprintln!("skipping: assets/yolo26n-face/vulkan/model.pte not built");
        return;
    };
    let Some(manifest_path) = yolo26n_face_asset("manifest.yaml") else {
        eprintln!("skipping: assets/yolo26n-face/manifest.yaml missing");
        return;
    };
    let Some(imgsz) = manifest_imgsz(&manifest_path) else {
        panic!("manifest.yaml missing imgsz");
    };

    let Some(ctx) = shared_vulkan_context() else {
        eprintln!("skipping: no Vulkan device");
        return;
    };

    let bytes = std::fs::read(&pte_path).expect("read vulkan model.pte");
    let meta = ProgramMetadata::from_bytes(&bytes).expect("parse vulkan program metadata");
    assert_detection_input_shape(&meta, imgsz);

    let backend = ExecuTorchBackend::new();
    let session = backend
        .load_model(
            &bytes,
            ExecuTorchBackendConfig::Vulkan {
                context: ctx,
                method: None,
            },
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
fn test_shared_vulkan_context_creation() {
    use infers_gpu::ash::vk::Handle;
    use infers_gpu::VulkanContext;
    use processing::GpuImageProcessor;
    use std::sync::Arc;

    let Ok(ctx) = VulkanContext::new(&Device::gpu(0)) else {
        eprintln!("skipping: no Vulkan device");
        return;
    };
    let ctx = Arc::new(ctx);
    let proc = GpuImageProcessor::new(Arc::clone(&ctx)).expect("processor");
    assert_eq!(
        proc.context().device_handle().as_raw(),
        ctx.device_handle().as_raw()
    );

    // Teardown order: drop processor before context (Arc handles this).
    drop(proc);
    drop(ctx);
}

#[test]
#[cfg(feature = "vulkan")]
fn test_prepare_gpu_inputs_avoids_read_to_cpu() {
    use infers_backend_executorch::gpu_input;
    use infers_core::{DataType, TensorBuffer, TensorShape};
    use infers_gpu::VulkanContext;
    use processing::GpuTensorBuffer;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    static READBACK: AtomicBool = AtomicBool::new(false);

    struct TrackingGpuBuffer {
        inner: GpuTensorBuffer,
    }

    impl std::fmt::Debug for TrackingGpuBuffer {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.inner.fmt(f)
        }
    }

    impl TensorBuffer for TrackingGpuBuffer {
        fn shape(&self) -> &TensorShape {
            self.inner.shape()
        }
        fn dtype(&self) -> DataType {
            self.inner.dtype()
        }
        fn device(&self) -> &infers_core::Device {
            self.inner.device()
        }
        fn as_any(&self) -> &dyn std::any::Any {
            // Downcast in prepare_inputs expects GpuTensorBuffer, not this wrapper.
            self.inner.as_any()
        }
        fn read_to_cpu(&self) -> Result<Box<dyn infers_core::AnyHostTensor>, infers_core::CoreError> {
            READBACK.store(true, Ordering::SeqCst);
            self.inner.read_to_cpu()
        }
        fn copy_to_device(
            &self,
            target: &infers_core::Device,
        ) -> Result<Box<dyn TensorBuffer>, infers_core::CoreError> {
            self.inner.copy_to_device(target)
        }
    }

    let Ok(ctx) = VulkanContext::new(&infers_core::Device::gpu(0)) else {
        eprintln!("skipping: no Vulkan device");
        return;
    };
    let ctx = Arc::new(ctx);
    let shape = TensorShape::new(vec![1, 64, 64, 3]).unwrap();
    let gpu = GpuTensorBuffer::new_zeros(Arc::clone(&ctx), shape.clone(), DataType::F32)
        .expect("test buffer");
    let tracked = TrackingGpuBuffer { inner: gpu };
    let input: &dyn TensorBuffer = &tracked;

    READBACK.store(false, Ordering::SeqCst);
    let plan = gpu_input::prepare_inputs(&[input], &ctx, None).expect("prepare_inputs");
    assert!(!READBACK.load(Ordering::SeqCst), "GpuTensorBuffer path must not read_to_cpu");
    assert_eq!(plan.tensor_ptrs.len(), 1);
    assert_eq!(plan.skip_staging_mask, 0);
}

#[test]
#[cfg(feature = "vulkan")]
fn test_vulkan_config_registers_or_reports_missing_backend() {
    let Some(ctx) = shared_vulkan_context() else {
        eprintln!("skipping: no Vulkan device");
        return;
    };
    let backend = ExecuTorchBackend::new();
    let result = backend.load_model(
        &[0u8; 16],
        ExecuTorchBackendConfig::Vulkan {
            context: ctx,
            method: None,
        },
    );
    assert!(result.is_err());
}
