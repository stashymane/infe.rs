use infers_core::{Device, ImageFormat, ImageInputBuffer, ProcessingOptions, Rotation};
use infers_test_utils::mock_hardware_buffer;
use platform_android::AHARDWAREBUFFER_FORMAT_R8G8B8_UNORM;
use processing::{CpuImageProcessor, FitMode, GpuImageProcessor};

#[test]
fn test_android_hardware_buffer_mock_and_metadata() {
    let raw_data = vec![128u8; 64 * 64 * 3];
    let handle = mock_hardware_buffer(
        64,
        64,
        ImageFormat::RGB888,
        raw_data.clone(),
        Device::gpu(0),
    );

    assert_eq!(handle.width(), 64);
    assert_eq!(handle.height(), 64);
    assert_eq!(handle.format(), ImageFormat::RGB888);
    assert_eq!(handle.device(), &Device::gpu(0));
    assert_eq!(handle.desc().width, 64);
    assert_eq!(handle.desc().height, 64);
    assert_eq!(handle.desc().format, AHARDWAREBUFFER_FORMAT_R8G8B8_UNORM);

    let vulkan_import = handle.as_vulkan_external_memory().unwrap();
    assert_eq!(vulkan_import.width, 64);
    assert_eq!(vulkan_import.height, 64);
    assert!(vulkan_import.supports_gpu_sampling());

    let locked = handle.lock_cpu_read().unwrap();
    assert_eq!(locked.as_slice().len(), 64 * 64 * 3);
}

#[test]
fn test_android_hardware_buffer_with_image_processors() {
    let mut raw_data = Vec::with_capacity(32 * 32 * 3);
    for i in 0..(32 * 32 * 3) {
        raw_data.push((i % 256) as u8);
    }

    let handle = mock_hardware_buffer(
        32,
        32,
        ImageFormat::RGB888,
        raw_data,
        Device::gpu(0),
    );

    let opts = ProcessingOptions {
        src_w: 32,
        src_h: 32,
        dest_w: 16,
        dest_h: 16,
        dest_format: ImageFormat::RGBF32,
        fit_mode: FitMode::STRETCH,
        rotation: Rotation::None,
        ..Default::default()
    };

    let cpu_proc = CpuImageProcessor::new();
    let cpu_out = cpu_proc.process(&handle, &opts).unwrap();
    assert_eq!(cpu_out.shape().dims(), &[1, 16, 16, 3]);

    let gpu_proc = GpuImageProcessor::new(&Device::gpu(0)).unwrap();
    let gpu_out = gpu_proc.process(&handle, &opts).unwrap();
    assert_eq!(gpu_out.shape().dims(), &[1, 16, 16, 3]);
    assert_eq!(gpu_out.device(), &Device::gpu(0));
}
