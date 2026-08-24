use infers_core::{
    Backend, CpuImageBuffer, CpuTensor, DataType, Device, DeviceKind, ImageFormat, TensorBuffer,
    TensorShape, CoreError,
};
use infers_test_utils::{MockBackend, MockDeviceTensor};

#[test]
fn test_device_creation_and_properties() {
    let cpu = Device::cpu();
    assert_eq!(cpu.kind, DeviceKind::Cpu);
    assert_eq!(cpu.id, 0);
    assert!(cpu.is_cpu());
    assert!(!cpu.is_gpu());

    let gpu = Device::gpu(1);
    assert_eq!(gpu.kind, DeviceKind::Gpu);
    assert_eq!(gpu.id, 1);
    assert!(gpu.is_gpu());
    assert_eq!(gpu.to_string(), "GPU:1");

    let npu = Device::npu(0);
    assert_eq!(npu.kind, DeviceKind::Npu);
    assert!(npu.is_npu());
    assert_eq!(npu.to_string(), "NPU:0");
}

#[test]
fn test_tensor_shape_validation() {
    let shape = TensorShape::new(vec![1, 3, 224, 224]).unwrap();
    assert_eq!(shape.dims(), &[1, 3, 224, 224]);
    assert_eq!(shape.rank(), 4);
    assert_eq!(shape.element_count(), 1 * 3 * 224 * 224);
    assert_eq!(shape.byte_size(DataType::F32), 1 * 3 * 224 * 224 * 4);
    assert_eq!(shape.byte_size(DataType::U8), 1 * 3 * 224 * 224 * 1);

    assert!(TensorShape::new(vec![]).is_err());
    assert!(TensorShape::new(vec![1, 0, 224]).is_err());
}

#[test]
fn test_cpu_tensor_operations() {
    let shape = TensorShape::from([2, 3]);
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = CpuTensor::from_f32(shape.clone(), data.clone()).unwrap();

    assert_eq!(tensor.shape(), &shape);
    assert_eq!(tensor.dtype(), DataType::F32);
    assert_eq!(tensor.as_slice(), &data);

    let host_box = tensor.read_to_cpu().unwrap();
    let f32_slice = host_box.as_slice_f32().unwrap();
    assert_eq!(f32_slice, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert!(host_box.as_slice_u8().is_err());
}

#[test]
fn test_opaque_device_tensor_residency_and_explicit_read() {
    let gpu = Device::gpu(0);
    let shape = TensorShape::from([1, 4]);
    let initial_values = vec![12.5f32, 25.0, 50.0, 100.0];

    let gpu_tensor = MockDeviceTensor::from_f32_slice(gpu.clone(), shape.clone(), &initial_values);
    assert_eq!(gpu_tensor.device(), &gpu);
    assert_eq!(gpu_tensor.shape(), &shape);
    assert_eq!(gpu_tensor.dtype(), DataType::F32);

    // Explicit readback
    let host_tensor = gpu_tensor.read_to_cpu().unwrap();
    let slice = host_tensor.as_slice_f32().unwrap();
    assert_eq!(slice, &initial_values);

    // Device copy to NPU
    let npu = Device::npu(0);
    let npu_tensor = gpu_tensor.copy_to_device(&npu).unwrap();
    assert_eq!(npu_tensor.device(), &npu);
    let npu_host = npu_tensor.read_to_cpu().unwrap();
    assert_eq!(npu_host.as_slice_f32().unwrap(), &initial_values);
}

#[test]
fn test_mock_backend_and_session_device_mismatch() {
    let gpu = Device::gpu(0);
    let backend = MockBackend::new("mock-backend");
    let mut session = backend.load_model(&[0u8; 16], &gpu).unwrap();

    // Passing CPU tensor to GPU session must return DeviceMismatch
    let cpu_tensor = CpuTensor::from_f32(
        TensorShape::from([1, 3, 224, 224]),
        vec![0.0f32; 1 * 3 * 224 * 224],
    )
    .unwrap();

    let result = session.run(&[&cpu_tensor]);
    match result {
        Err(CoreError::DeviceMismatch { expected, actual }) => {
            assert_eq!(expected, gpu);
            assert_eq!(actual, Device::cpu());
        }
        other => panic!("Expected DeviceMismatch error, got {:?}", other),
    }

    // Passing matching GPU tensor must succeed
    let gpu_tensor = MockDeviceTensor::from_f32_slice(
        gpu.clone(),
        TensorShape::from([1, 3, 224, 224]),
        &vec![0.0f32; 1 * 3 * 224 * 224],
    );

    let outputs = session.run(&[&gpu_tensor]).unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].device(), &gpu);
    let out_host = outputs[0].read_to_cpu().unwrap();
    assert_eq!(out_host.as_slice_f32().unwrap().len(), 4);
}

#[test]
fn test_cpu_image_buffer() {
    let raw = vec![255u8; 64 * 64 * 3];
    let img = CpuImageBuffer::new(64, 64, ImageFormat::RGB888, raw).unwrap();
    assert_eq!(img.width(), 64);
    assert_eq!(img.height(), 64);
    assert_eq!(img.format(), ImageFormat::RGB888);
}
