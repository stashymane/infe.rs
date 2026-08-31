use infers_bindings::*;
use infers_core::Backend;
use infers_test_utils::MockBackend;

mod common;

fn f32_le_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f32_from_le_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn test_uniffi_device_creation() {
    let cpu = create_cpu_device();
    assert_eq!(cpu.kind, DeviceKind::Cpu);
    assert_eq!(cpu.id, 0);
    assert_eq!(cpu.name, "CPU");

    let gpu = create_gpu_device(1);
    assert_eq!(gpu.kind, DeviceKind::Gpu);
    assert_eq!(gpu.id, 1);
    assert_eq!(gpu.name, "GPU:1");

    let npu = create_npu_device(2);
    assert_eq!(npu.kind, DeviceKind::Npu);
    assert_eq!(npu.id, 2);
    assert_eq!(npu.name, "NPU:2");
}

#[test]
fn test_uniffi_tensor_buffers() {
    let shape = TensorShape {
        dims: vec![1, 3, 2, 2],
    };
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let bytes = f32_le_bytes(&data);

    let tensor =
        create_tensor_from_bytes(shape.clone(), DataType::F32, bytes).expect("Failed to create tensor");
    assert_eq!(tensor.shape(), shape);
    assert_eq!(tensor.dtype(), DataType::F32);
    assert_eq!(tensor.byte_size(), 48);

    let read_back = f32_from_le_bytes(&tensor.read_to_cpu_bytes().expect("Failed to read bytes"));
    assert_eq!(read_back, data);

    // Transport endianness is little-endian on all supported hosts.
    assert_eq!(1.0f32.to_le_bytes(), 1.0f32.to_ne_bytes());

    let cpu_dev = create_cpu_device();
    let copied = tensor
        .copy_to_device(cpu_dev, None)
        .expect("Copy to CPU should succeed");
    assert_eq!(
        f32_from_le_bytes(&copied.read_to_cpu_bytes().unwrap()),
        data
    );
}

#[test]
fn test_uniffi_image_processor_cpu() {
    let processor = create_cpu_image_processor();
    assert_eq!(processor.device().kind, DeviceKind::Cpu);

    // 4x4 RGB image (48 bytes)
    let raw_bytes: Vec<u8> = (0..48).map(|v| (v * 5) as u8).collect();

    let options = ProcessingOptions {
        src_w: 4,
        src_h: 4,
        crop_x: 0,
        crop_y: 0,
        crop_w: 4,
        crop_h: 4,
        dest_w: 2,
        dest_h: 2,
        src_format: ImageFormat::Rgb888,
        dest_format: ImageFormat::Rgbf32,
        fit_mode: FitMode::Stretch,
        rotation: Rotation::None,
        dest_layout: TensorLayout::Nchw,
    };

    let processed = processor
        .process_bytes(raw_bytes, 4, 4, ImageFormat::Rgb888, options)
        .expect("CPU image processing failed");

    assert_eq!(
        processed.shape(),
        TensorShape {
            dims: vec![1, 3, 2, 2]
        }
    );
    assert_eq!(processed.dtype(), DataType::F32);

    let output_floats =
        f32_from_le_bytes(&processed.read_to_cpu_bytes().expect("Failed to read processed floats"));
    assert_eq!(output_floats.len(), 12);
}

#[test]
fn test_uniffi_mock_backend_and_session() {
    let backend = MockBackend::new("mock_backend");
    let devices = backend.available_devices();
    assert!(!devices.is_empty());

    let cpu = create_cpu_device();
    let session = common::load_mock_session(vec![0x01, 0x02, 0x03], cpu.clone());

    assert_eq!(session.device(), cpu);

    let input_tensor = create_tensor_from_bytes(
        TensorShape {
            dims: vec![1, 3, 224, 224],
        },
        DataType::F32,
        f32_le_bytes(&vec![0.5f32; 3 * 224 * 224]),
    )
    .expect("Failed to create input tensor");

    let outputs = session
        .run(vec![input_tensor])
        .expect("Mock session run failed");

    assert_eq!(outputs.len(), 1);
    assert_eq!(
        outputs[0].shape(),
        TensorShape {
            dims: vec![1, 4]
        }
    );
    assert_eq!(outputs[0].dtype(), DataType::F32);

    let result = f32_from_le_bytes(&outputs[0].read_to_cpu_bytes().unwrap());
    assert_eq!(result, vec![10.0, 20.0, 100.0, 150.0]);
}

#[test]
fn test_uniffi_executorch_backend_error_handling() {
    let backend = FfiBackend::new();
    let devices = backend.available_devices();
    assert!(!devices.is_empty());

    // Invalid model bytes should return ModelLoadFailed
    let invalid_bytes = vec![0xDE, 0xAD, 0xBE, 0xEF];
    let err = backend.load_model(
        invalid_bytes,
        BackendConfig::Xnnpack {
            num_threads: 1,
            method: None,
        },
    );
    assert!(err.is_err());
    match err.unwrap_err() {
        InfersError::ModelLoadFailed { reason } => {
            assert!(!reason.is_empty());
        }
        other => panic!("Unexpected error type: {:?}", other),
    }
}
