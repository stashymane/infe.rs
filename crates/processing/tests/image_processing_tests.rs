use infers_core::{
    CpuImageBuffer, DataType, Device, ImageFormat, ProcessingOptions, Rotation,
};
use processing_core::TensorLayout;
#[cfg(feature = "vulkan")]
use infers_gpu::VulkanContext;
use processing::{CpuImageProcessor, FitMode};
#[cfg(feature = "vulkan")]
use processing::GpuImageProcessor;
use std::sync::Arc;

fn create_test_pattern_image(width: u32, height: u32) -> (Vec<u8>, CpuImageBuffer) {
    let mut data = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let r = ((x * 255) / width.max(1)) as u8;
            let g = ((y * 255) / height.max(1)) as u8;
            let b = (((x + y) * 255) / (width + height).max(1)) as u8;
            data.push(r);
            data.push(g);
            data.push(b);
        }
    }
    let buf = CpuImageBuffer::new(width, height, ImageFormat::Rgb888, data.clone()).unwrap();
    (data, buf)
}

#[test]
fn test_cpu_processor_stretch_rgb888() {
    let (_, input) = create_test_pattern_image(64, 64);
    let processor = CpuImageProcessor::new();

    let opts = ProcessingOptions {
        src_w: 64,
        src_h: 64,
        dest_w: 32,
        dest_h: 32,
        dest_format: ImageFormat::Rgb888,
        fit_mode: FitMode::Stretch,
        rotation: Rotation::None,
        ..Default::default()
    };

    let tensor_buf = processor.process(&input, &opts).unwrap();
    assert_eq!(tensor_buf.shape().dims(), &[1, 32, 32, 3]);
    assert_eq!(tensor_buf.dtype(), DataType::U8);
    assert_eq!(tensor_buf.device(), &Device::cpu());

    let host = tensor_buf.read_to_cpu().unwrap();
    let bytes = host.as_slice_u8().unwrap();
    assert_eq!(bytes.len(), 32 * 32 * 3);
}

#[test]
fn test_cpu_processor_contain_rgbf32() {
    let (_, input) = create_test_pattern_image(100, 50);
    let processor = CpuImageProcessor::new();

    let opts = ProcessingOptions {
        src_w: 100,
        src_h: 50,
        dest_w: 64,
        dest_h: 64,
        dest_format: ImageFormat::Rgbf32,
        dest_layout: TensorLayout::default_for_dest_format(ImageFormat::Rgbf32),
        fit_mode: FitMode::Contain,
        rotation: Rotation::None,
        ..Default::default()
    };

    let tensor_buf = processor.process(&input, &opts).unwrap();
    assert_eq!(tensor_buf.shape().dims(), &[1, 3, 64, 64]);
    assert_eq!(tensor_buf.dtype(), DataType::F32);

    let host = tensor_buf.read_to_cpu().unwrap();
    let f32_slice = host.as_slice_f32().unwrap();
    assert_eq!(f32_slice.len(), 64 * 64 * 3);

    // Assert normalized values are between 0.0 and 1.0
    for &val in f32_slice {
        assert!((0.0..=1.0).contains(&val), "Value {} is out of [0.0, 1.0]", val);
    }
}

#[test]
fn test_cpu_processor_rotations() {
    let (_, input) = create_test_pattern_image(60, 40);
    let processor = CpuImageProcessor::new();

    for rot in [
        Rotation::None,
        Rotation::Rot90,
        Rotation::Rot180,
        Rotation::Rot270,
    ] {
        let opts = ProcessingOptions {
            src_w: 60,
            src_h: 40,
            dest_w: 48,
            dest_h: 48,
            dest_format: ImageFormat::Rgb888,
            fit_mode: FitMode::Stretch,
            rotation: rot,
            ..Default::default()
        };

        let tensor_buf = processor.process(&input, &opts).unwrap();
        assert_eq!(tensor_buf.shape().dims(), &[1, 48, 48, 3]);
        let host = tensor_buf.read_to_cpu().unwrap();
        assert_eq!(host.as_slice_u8().unwrap().len(), 48 * 48 * 3);
    }
}

#[test]
#[cfg(feature = "vulkan")]
fn test_gpu_process_outputs() {
    let (_, input) = create_test_pattern_image(80, 60);
    let device = Device::gpu(0);
    let context = match VulkanContext::new(&device) {
        Ok(ctx) => Arc::new(ctx),
        Err(err) => {
            eprintln!("skipping GPU process test: {err}");
            return;
        }
    };
    let gpu_processor = match GpuImageProcessor::new(context) {
        Ok(proc) => proc,
        Err(err) => {
            eprintln!("skipping GPU process test: {err}");
            return;
        }
    };

    let test_cases = vec![
        ProcessingOptions {
            src_w: 80,
            src_h: 60,
            crop_x: 10,
            crop_y: 10,
            crop_w: 40,
            crop_h: 30,
            dest_w: 32,
            dest_h: 32,
            dest_format: ImageFormat::Rgb888,
            fit_mode: FitMode::Stretch,
            rotation: Rotation::None,
            ..Default::default()
        },
        ProcessingOptions {
            src_w: 80,
            src_h: 60,
            crop_x: 0,
            crop_y: 0,
            crop_w: 80,
            crop_h: 60,
            dest_w: 32,
            dest_h: 32,
            dest_format: ImageFormat::Rgbf32,
            dest_layout: TensorLayout::default_for_dest_format(ImageFormat::Rgbf32),
            fit_mode: FitMode::Contain,
            rotation: Rotation::Rot90,
            ..Default::default()
        },
        ProcessingOptions {
            src_w: 80,
            src_h: 60,
            crop_x: 0,
            crop_y: 0,
            crop_w: 80,
            crop_h: 60,
            dest_w: 32,
            dest_h: 32,
            dest_format: ImageFormat::Rgb888,
            fit_mode: FitMode::Crop,
            rotation: Rotation::Rot180,
            ..Default::default()
        },
    ];

    for opts in test_cases {
        let gpu_buf = gpu_processor.process(&input, &opts).unwrap();
        let expected_dims = match opts.dest_format {
            ImageFormat::Rgbf32 => &[1, 3, 32, 32],
            _ => &[1, 32, 32, 3],
        };
        assert_eq!(gpu_buf.shape().dims(), expected_dims);
        assert_eq!(gpu_buf.device(), &Device::gpu(0));

        let gpu_host = gpu_buf.read_to_cpu().unwrap();
        if opts.dest_format == ImageFormat::Rgb888 {
            assert_eq!(gpu_buf.dtype(), DataType::U8);
            let bytes = gpu_host.as_slice_u8().unwrap();
            assert_eq!(bytes.len(), 32 * 32 * 3);
            assert!(bytes.iter().any(|&b| b > 0), "expected non-zero RGB888 output");
        } else {
            assert_eq!(gpu_buf.dtype(), DataType::F32);
            let vals = gpu_host.as_slice_f32().unwrap();
            assert_eq!(vals.len(), 32 * 32 * 3);
            for &v in vals {
                assert!((0.0..=1.0).contains(&v), "value {v} out of [0, 1]");
            }
            assert!(vals.iter().any(|&v| v > 0.0), "expected non-zero RGBF32 output");
        }
    }
}
