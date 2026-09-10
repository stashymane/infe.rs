#![cfg(feature = "vulkan")]

use infers_gpu::Vulkan;
use infers_processing::{FitMode, GpuImageProcessor, ImageFormat, ProcessingOptions};

#[test]
fn test_gpu_image_processor_loads_shaders() {
    let Ok(vulkan) = Vulkan::new(0) else {
        eprintln!("skipping: no Vulkan device");
        return;
    };
    GpuImageProcessor::new(vulkan).expect("SPIR-V shaders must load into pipelines");
}

#[test]
fn test_processing_options_enums() {
    assert_eq!(FitMode::Stretch as u32, 0);
    assert_eq!(FitMode::Contain as u32, 1);
    assert_eq!(FitMode::Crop as u32, 2);

    assert_eq!(ImageFormat::Rgb888 as u32, 0);
    assert_eq!(ImageFormat::Rgbf32 as u32, 1);
    assert_eq!(ImageFormat::Nv12 as u32, 2);
    assert_eq!(ImageFormat::I420 as u32, 3);
}

#[test]
fn test_processing_options_crop() {
    let opts_no_crop = ProcessingOptions {
        src_w: 1920,
        src_h: 1080,
        crop_x: 0,
        crop_y: 0,
        crop_w: 0,
        crop_h: 0,
        dest_w: 224,
        dest_h: 224,
        dest_format: ImageFormat::Rgbf32,
        fit_mode: FitMode::Contain,
        rotation_degrees: 0.0,
        ..Default::default()
    };

    assert_eq!(opts_no_crop.effective_crop(), (0, 0, 1920, 1080));

    let opts_with_crop = ProcessingOptions {
        src_w: 1920,
        src_h: 1080,
        crop_x: 100,
        crop_y: 50,
        crop_w: 400,
        crop_h: 300,
        dest_w: 224,
        dest_h: 224,
        dest_format: ImageFormat::Rgb888,
        fit_mode: FitMode::Crop,
        rotation_degrees: 90.0,
        ..Default::default()
    };

    assert_eq!(opts_with_crop.effective_crop(), (100, 50, 400, 300));
}
