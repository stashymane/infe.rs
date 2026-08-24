use infers_bindings::*;
use infers_core::Backend;
use infers_test_utils::MockBackend;

mod common;

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

    let tensor = create_tensor_from_f32(shape.clone(), data.clone()).expect("Failed to create tensor");
    assert_eq!(tensor.shape(), shape);
    assert_eq!(tensor.dtype(), DataType::F32);
    assert_eq!(tensor.byte_size(), 48);

    let read_back = tensor.read_to_cpu_f32().expect("Failed to read f32 tensor");
    assert_eq!(read_back, data);

    // Test copy to CPU device
    let cpu_dev = create_cpu_device();
    let copied = tensor.copy_to_device(cpu_dev).expect("Copy to CPU should succeed");
    assert_eq!(copied.read_to_cpu_f32().unwrap(), data);
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
        dest_format: ImageFormat::Rgbf32,
        fit_mode: FitMode::Stretch,
        rotation: Rotation::None,
    };

    let processed = processor
        .process_bytes(raw_bytes, 4, 4, ImageFormat::Rgb888, options)
        .expect("CPU image processing failed");

    assert_eq!(
        processed.shape(),
        TensorShape {
            dims: vec![1, 2, 2, 3]
        }
    );
    assert_eq!(processed.dtype(), DataType::F32);

    let output_floats = processed.read_to_cpu_f32().expect("Failed to read processed floats");
    assert_eq!(output_floats.len(), 12);
}

#[test]
fn test_uniffi_hardware_buffer_mock() {
    let cpu = create_cpu_device();
    let data = vec![255u8; 16 * 16 * 3];
    let hb = common::create_mock_hardware_buffer(16, 16, ImageFormat::Rgb888, data.clone(), cpu);

    assert_eq!(hb.width(), 16);
    assert_eq!(hb.height(), 16);

    let locked = hb.lock_cpu().expect("Lock CPU failed");
    assert_eq!(locked.len(), 16 * 16 * 3);
    assert_eq!(locked, data);

    let processor = create_cpu_image_processor();
    let options = ProcessingOptions {
        src_w: 16,
        src_h: 16,
        crop_x: 2,
        crop_y: 2,
        crop_w: 8,
        crop_h: 8,
        dest_w: 4,
        dest_h: 4,
        dest_format: ImageFormat::Rgb888,
        fit_mode: FitMode::Crop,
        rotation: Rotation::R90deg,
    };

    let out_tensor = processor
        .process_hardware_buffer(hb, options)
        .expect("Process mock hardware buffer failed");

    assert_eq!(
        out_tensor.shape(),
        TensorShape {
            dims: vec![1, 4, 4, 3]
        }
    );
    assert_eq!(out_tensor.dtype(), DataType::U8);
    let bytes = out_tensor.read_to_cpu_u8().unwrap();
    assert_eq!(bytes.len(), 4 * 4 * 3);
}

#[test]
fn test_uniffi_mock_backend_and_session() {
    let backend = MockBackend::new("mock_backend");
    let devices = backend.available_devices();
    assert!(!devices.is_empty());

    let cpu = create_cpu_device();
    let session = common::load_mock_session(vec![0x01, 0x02, 0x03], cpu.clone());

    assert_eq!(session.device(), cpu);

    let input_tensor = create_tensor_from_f32(
        TensorShape {
            dims: vec![1, 3, 224, 224],
        },
        vec![0.5f32; 1 * 3 * 224 * 224],
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

    let result = outputs[0].read_to_cpu_f32().unwrap();
    assert_eq!(result, vec![10.0, 20.0, 100.0, 150.0]);
}

#[test]
fn test_uniffi_executorch_backend_error_handling() {
    let backend = ExecuTorchBackend::new();
    let devices = backend.available_devices();
    assert!(!devices.is_empty());

    let cpu = create_cpu_device();
    // Invalid model bytes should return ModelLoadFailed
    let invalid_bytes = vec![0xDE, 0xAD, 0xBE, 0xEF];
    let err = backend.load_model(invalid_bytes, cpu);
    assert!(err.is_err());
    match err.unwrap_err() {
        InfersError::ModelLoadFailed { message } => {
            assert!(!message.is_empty());
        }
        other => panic!("Unexpected error type: {:?}", other),
    }
}
