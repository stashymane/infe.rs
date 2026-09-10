#![cfg(target_os = "android")]

use infers_core::{
    DeviceInfo, DeviceKind, HardwareImage, ImageFormat, ProcessingOptions,
};
use infers_platform_android::{
    AndroidHardwareBufferHandle, AHardwareBuffer_Desc, AHardwareBuffer_allocate,
    AHARDWAREBUFFER_FORMAT_R8G8B8_UNORM, AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN,
    AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN, AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE,
};
use infers_processing::{
    CpuImageProcessor, DeferredCpuProcessExt, DeferredVulkanProcessExt, FitMode,
    GpuImageProcessor, TensorLayout, Vulkan, VulkanImage,
};
use std::sync::Arc;

fn gpu0_info() -> DeviceInfo {
    DeviceInfo {
        kind: DeviceKind::Gpu,
        id: 0,
        name: "GPU:0".to_string(),
    }
}

fn allocate_rgb888_buffer(
    width: u32,
    height: u32,
    device: DeviceInfo,
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
    let device = gpu0_info();
    let handle = allocate_rgb888_buffer(64, 64, device.clone());

    assert_eq!(handle.width(), 64);
    assert_eq!(handle.height(), 64);
    assert_eq!(handle.format(), ImageFormat::Rgb888);
    assert_eq!(handle.device_info(), &device);
    assert_eq!(handle.desc().width, 64);
    assert_eq!(handle.desc().height, 64);
    assert_eq!(handle.desc().format, AHARDWAREBUFFER_FORMAT_R8G8B8_UNORM);
    assert!(handle.supports_gpu_sampling());

    let locked = handle.lock_cpu_read().unwrap();
    assert!(!locked.as_slice().is_empty());
}

#[test]
fn test_from_owned_takes_allocate_reference() {
    let device = gpu0_info();
    let desc = AHardwareBuffer_Desc {
        width: 8,
        height: 8,
        layers: 1,
        format: AHARDWAREBUFFER_FORMAT_R8G8B8_UNORM,
        usage: AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN | AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN,
        stride: 0,
        rfu0: 0,
        rfu1: 0,
    };
    let handle = unsafe {
        let mut ptr = std::ptr::null_mut();
        let status = AHardwareBuffer_allocate(&desc, &mut ptr);
        assert_eq!(status, 0, "AHardwareBuffer_allocate failed: {status}");
        // from_owned must not acquire again; dropping the handle releases the
        // allocate +1 exactly once.
        AndroidHardwareBufferHandle::from_owned(ptr, device).expect("from_owned")
    };
    assert_eq!(handle.width(), 8);
    assert_eq!(handle.height(), 8);
}

#[test]
fn test_from_borrowed_acquires_independent_reference() {
    let device = gpu0_info();
    let allocated = allocate_rgb888_buffer(8, 8, device.clone());
    let ptr = allocated.raw_ptr();
    // Simulate a borrowed view (as fromHardwareBuffer): acquire a second +1.
    let borrowed = AndroidHardwareBufferHandle::from_borrowed(ptr, device).expect("from_borrowed");
    assert_eq!(borrowed.width(), 8);
    // Dropping `borrowed` releases only the acquired +1; `allocated` still owns
    // the allocate reference.
    drop(borrowed);
    assert_eq!(allocated.width(), 8);
}

#[test]
fn test_from_owned_null_pointer() {
    let err = AndroidHardwareBufferHandle::from_owned(std::ptr::null_mut(), gpu0_info())
        .expect_err("null pointer must fail");
    assert!(matches!(
        err,
        infers_platform_android::AndroidPlatformError::NullBufferPointer
    ));
}

#[test]
fn test_from_borrowed_null_pointer() {
    let err = AndroidHardwareBufferHandle::from_borrowed(std::ptr::null_mut(), gpu0_info())
        .expect_err("null pointer must fail");
    assert!(matches!(
        err,
        infers_platform_android::AndroidPlatformError::NullBufferPointer
    ));
}

#[test]
fn test_to_vulkan_consumes_handle_and_keeps_sampled_image() {
    let device = gpu0_info();
    let handle = allocate_rgb888_buffer(16, 16, device.clone());
    let context = match infers_platform_android::create_vulkan_context(&device) {
        Ok(ctx) => Arc::new(ctx),
        Err(err) => {
            eprintln!("skipping Android GPU import ownership test: {err}");
            return;
        }
    };
    let sampled = match handle.to_vulkan(Arc::clone(&context)) {
        Ok(img) => img,
        Err(err) => {
            eprintln!("skipping Android GPU import ownership test: {err}");
            return;
        }
    };
    // `handle` was moved into to_vulkan and dropped after Vulkan acquired its
    // own AHB reference; the sampled image must still be usable.
    assert_eq!(sampled.width(), 16);
    assert_eq!(sampled.height(), 16);
    drop(sampled);
}

#[test]
fn test_on_moves_into_deferred_import() {
    let device = gpu0_info();
    let handle = allocate_rgb888_buffer(16, 16, device.clone());
    let context = match infers_platform_android::create_vulkan_context(&device) {
        Ok(ctx) => Arc::new(ctx),
        Err(err) => {
            eprintln!("skipping Android deferred import test: {err}");
            return;
        }
    };
    let vulkan = Vulkan::from_context(context);
    let gpu_proc = match GpuImageProcessor::new(vulkan.clone()) {
        Ok(proc) => proc,
        Err(err) => {
            eprintln!("skipping Android deferred import test: {err}");
            return;
        }
    };
    let opts = ProcessingOptions {
        src_w: 16,
        src_h: 16,
        dest_w: 8,
        dest_h: 8,
        dest_format: ImageFormat::Rgbf32,
        dest_layout: TensorLayout::default_for_dest_format(ImageFormat::Rgbf32),
        fit_mode: FitMode::Stretch,
        rotation_degrees: 0.0,
        ..Default::default()
    };
    // Consumes `handle` into Deferred; import + AHB release happen at materialize.
    let out = handle
        .on(&vulkan)
        .process(&gpu_proc, &opts)
        .unwrap()
        .materialize()
        .unwrap();
    assert_eq!(out.shape().dims(), &[1, 3, 8, 8]);
}

#[test]
fn test_android_hardware_buffer_with_image_processors() {
    let device = gpu0_info();
    let handle = allocate_rgb888_buffer(32, 32, device.clone());
    let cpu_img = HardwareImage::new(
        handle.width(),
        handle.height(),
        handle.format(),
        handle.copy_cpu_packed().unwrap(),
    )
    .unwrap();

    let opts = ProcessingOptions {
        src_w: 32,
        src_h: 32,
        dest_w: 16,
        dest_h: 16,
        dest_format: ImageFormat::Rgbf32,
        dest_layout: TensorLayout::default_for_dest_format(ImageFormat::Rgbf32),
        fit_mode: FitMode::Stretch,
        rotation_degrees: 0.0,
        ..Default::default()
    };

    let cpu_proc = CpuImageProcessor::new();
    let cpu_out = cpu_img
        .on_cpu()
        .process(&cpu_proc, &opts)
        .unwrap()
        .materialize()
        .unwrap();
    assert_eq!(cpu_out.shape().dims(), &[1, 3, 16, 16]);

    let context = match infers_platform_android::create_vulkan_context(&device) {
        Ok(ctx) => Arc::new(ctx),
        Err(err) => {
            eprintln!("skipping Android GPU processor test: {err}");
            return;
        }
    };
    let vulkan = Vulkan::from_context(Arc::clone(&context));
    let gpu_proc = match GpuImageProcessor::new(vulkan.clone()) {
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
    let gpu_image = VulkanImage::from_sampled(sampled);
    let gpu_out = gpu_proc
        .process(&gpu_image, &opts)
        .unwrap()
        .materialize()
        .unwrap();
    assert_eq!(gpu_out.shape().dims(), &[1, 3, 16, 16]);
    assert_eq!(gpu_out.device().info(), vulkan.info());
}
