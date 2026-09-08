use infers_bindings::*;

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
fn test_cpu_device_info() {
    let cpu = CpuDevice::new();
    let info = cpu.info();
    assert_eq!(info.kind, DeviceKind::Cpu);
    assert_eq!(info.id, 0);
    assert_eq!(info.name, "CPU");
}

#[test]
fn test_cpu_tensor_buffers() {
    let shape = TensorShape {
        dims: vec![1, 3, 2, 2],
    };
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let bytes = f32_le_bytes(&data);

    let tensor =
        create_cpu_tensor_from_bytes(shape.clone(), DataType::F32, bytes).expect("Failed to create tensor");
    assert_eq!(tensor.shape(), shape);
    assert_eq!(tensor.dtype(), DataType::F32);
    assert_eq!(tensor.byte_size(), 48);

    let read_back = f32_from_le_bytes(&tensor.read_bytes().expect("Failed to read bytes"));
    assert_eq!(read_back, data);
}

#[test]
fn test_cpu_image_processor() {
    let processor = CpuImageProcessor::new();
    assert_eq!(processor.device_info().kind, DeviceKind::Cpu);

    let raw_bytes: Vec<u8> = (0..48).map(|v| (v * 5) as u8).collect();
    let hardware =
        create_hardware_image(4, 4, ImageFormat::Rgb888, raw_bytes).expect("hardware image");

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
        rotation_degrees: 0.0,
        dest_layout: TensorLayout::Nchw,
    };

    let deferred = hardware.on_cpu();
    let pending = deferred.process(processor, options).expect("process");
    let out = pending.materialize().expect("materialize");
    assert_eq!(out.shape().dims, vec![1, 3, 2, 2]);
    assert_eq!(out.dtype(), DataType::F32);
}

#[test]
fn test_ffi_backend_construct() {
    let _backend = FfiBackend::new();
}
