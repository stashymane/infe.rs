use infers_core::{
    Cpu, DataType, HardwareImage, HostTensor, ImageFormat, Tensor, TensorShape,
};

#[test]
fn test_tensor_shape_validation() {
    let shape = TensorShape::new(vec![1, 3, 224, 224]).unwrap();
    assert_eq!(shape.dims(), &[1, 3, 224, 224]);
    assert_eq!(shape.rank(), 4);
    assert_eq!(shape.element_count(), 3 * 224 * 224);
    assert_eq!(shape.byte_size(DataType::F32), 3 * 224 * 224 * 4);
    assert_eq!(shape.byte_size(DataType::U8), 3 * 224 * 224);

    assert!(TensorShape::new(vec![]).is_err());
    assert!(TensorShape::new(vec![1, 0, 224]).is_err());
}

#[test]
fn test_cpu_tensor_roundtrip() {
    let shape = TensorShape::new(vec![2, 2]).unwrap();
    let host = HostTensor::from_f32(shape.clone(), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let tensor = Tensor::from_host(&Cpu, &host).unwrap();
    assert_eq!(tensor.shape(), &shape);
    assert_eq!(tensor.dtype(), DataType::F32);
    let back = tensor.read_to_host().unwrap();
    assert_eq!(back.as_slice_f32().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_host_image_validation() {
    let bytes = vec![0u8; 4 * 4 * 3];
    let img = HardwareImage::new(4, 4, ImageFormat::Rgb888, bytes).unwrap();
    assert_eq!(img.width(), 4);
    assert_eq!(img.height(), 4);
    assert_eq!(img.format(), ImageFormat::Rgb888);

    let bad = HardwareImage::new(4, 4, ImageFormat::Rgb888, vec![0u8; 10]);
    assert!(bad.is_err());
}

#[test]
fn test_host_image_empty_write() {
    let mut img = HardwareImage::empty(2, 2, ImageFormat::Rgb888).unwrap();
    assert_eq!(img.as_bytes().len(), 12);
    assert!(img.as_bytes().iter().all(|&b| b == 0));

    let filled: Vec<u8> = (0..12).map(|v| v as u8).collect();
    img.write_bytes(&filled).unwrap();
    assert_eq!(img.as_bytes(), filled.as_slice());

    assert!(img.write_bytes(&[1, 2, 3]).is_err());
}

#[test]
fn test_cpu_to_cpu_adopt() {
    let shape = TensorShape::new(vec![4]).unwrap();
    let host = HostTensor::from_f32(shape, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let tensor = Tensor::from_host(&Cpu, &host).unwrap();
    let adopted = tensor.to_device(&Cpu).unwrap();
    assert_eq!(adopted.read_to_host().unwrap().as_slice_f32().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
}
