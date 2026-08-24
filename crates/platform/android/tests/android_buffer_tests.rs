#![cfg(target_os = "android")]

use infers_core::{CpuImageBuffer, Device, ImageFormat, ImageInputBuffer, ProcessingOptions, Rotation};
use platform_android::{
    AndroidHardwareBufferHandle, AHardwareBuffer_Desc, AHardwareBuffer_allocate,
    AHARDWAREBUFFER_FORMAT_R8G8B8_UNORM, AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN,
    AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN, AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE,
};
use processing::{CpuImageProcessor, FitMode, GpuImageProcessor};
use std::sync::Arc;

fn allocate_rgb888_buffer(
    width: u32,
    height: u32,
    device: Device,
) -> AndroidHardwareBufferHandle {
    let desc = AHardwareBuffer_Desc {
        width,
        height,
        layers: 1,
        format: AHARDWAREBUFFER_FORMAT_R8G8B8_UNORM,
        usage: AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE
            | AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN
            | AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN,
        stride: 0,
        rfu0: 0,
        rfu1: 0,
    };

    unsafe {
        let mut ptr = std::ptr::null_mut();
        let status = AHardwareBuffer_allocate(&desc, &mut ptr);
        assert_eq!(status, 0, "AHardwareBuffer_allocate failed: {status}");
        AndroidHardwareBufferHandle::from_allocated(ptr, device)
            .expect("from_allocated should succeed for a non-null buffer")
    }
}

#[test]
fn test_android_hardware_buffer_metadata() {
    let handle = allocate_rgb888_buffer(64, 64, Device::gpu(0));

    assert_eq!(handle.width(), 64);
    assert_eq!(handle.height(), 64);
    assert_eq!(handle.format(), ImageFormat::RGB888);
    assert_eq!(handle.device(), &Device::gpu(0));
    assert_eq!(handle.desc().width, 64);
    assert_eq!(handle.desc().height, 64);
    assert_eq!(handle.desc().format, AHARDWAREBUFFER_FORMAT_R8G8B8_UNORM);
    assert!(handle.supports_gpu_sampling());

    let locked = handle.lock_cpu_read().unwrap();
    assert!(!locked.as_slice().is_empty());
}

#[test]
fn test_android_hardware_buffer_with_image_processors() {
    let handle = allocate_rgb888_buffer(32, 32, Device::gpu(0));
    let locked = handle.lock_cpu_read().unwrap();
    let cpu_img = CpuImageBuffer::new(
        handle.width(),
        handle.height(),
        handle.format(),
        locked.as_slice().to_vec(),
    )
    .unwrap();
    drop(locked);

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
    let cpu_out = cpu_proc.process(&cpu_img, &opts).unwrap();
    assert_eq!(cpu_out.shape().dims(), &[1, 16, 16, 3]);

    let context = match platform_android::create_vulkan_context(&Device::gpu(0)) {
        Ok(ctx) => Arc::new(ctx),
        Err(err) => {
            eprintln!("skipping Android GPU processor test: {err}");
            return;
        }
    };
    let gpu_proc = match GpuImageProcessor::new(Arc::clone(&context)) {
        Ok(proc) => proc,
        Err(err) => {
            eprintln!("skipping Android GPU processor test: {err}");
            return;
        }
    };
    let sampled = match handle.to_vulkan(context) {
        Ok(img) => img,
        Err(err) => {
            eprintln!("skipping Android GPU import: {err}");
            return;
        }
    };
    let gpu_out = gpu_proc.process(&sampled, &opts).unwrap();
    assert_eq!(gpu_out.shape().dims(), &[1, 16, 16, 3]);
    assert_eq!(gpu_out.device(), &Device::gpu(0));
}
