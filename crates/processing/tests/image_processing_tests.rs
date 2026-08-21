use infers_core::{
    CpuImageBuffer, DataType, Device, ImageFormat, ProcessingOptions, Rotation,
};
use processing::{CpuImageProcessor, FitMode, GpuImageProcessor};

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
    let buf = CpuImageBuffer::new(width, height, ImageFormat::RGB888, data.clone()).unwrap();
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
        dest_format: ImageFormat::RGB888,
        fit_mode: FitMode::STRETCH,
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
        dest_format: ImageFormat::RGBF32,
        fit_mode: FitMode::CONTAIN,
        rotation: Rotation::None,
        ..Default::default()
    };

    let tensor_buf = processor.process(&input, &opts).unwrap();
    assert_eq!(tensor_buf.shape().dims(), &[1, 64, 64, 3]);
    assert_eq!(tensor_buf.dtype(), DataType::F32);

    let host = tensor_buf.read_to_cpu().unwrap();
    let f32_slice = host.as_slice_f32().unwrap();
    assert_eq!(f32_slice.len(), 64 * 64 * 3);

    // Assert normalized values are between 0.0 and 1.0
    for &val in f32_slice {
        assert!(val >= 0.0 && val <= 1.0, "Value {} is out of [0.0, 1.0]", val);
    }
}

#[test]
fn test_cpu_processor_rotations() {
    let (_, input) = create_test_pattern_image(60, 40);
    let processor = CpuImageProcessor::new();

    for rot in [
        Rotation::None,
        Rotation::R90DEG,
        Rotation::R180DEG,
        Rotation::R270DEG,
    ] {
        let opts = ProcessingOptions {
            src_w: 60,
            src_h: 40,
            dest_w: 48,
            dest_h: 48,
            dest_format: ImageFormat::RGB888,
            fit_mode: FitMode::STRETCH,
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
fn test_cpu_and_gpu_numerical_parity() {
    let (_, input) = create_test_pattern_image(80, 60);
    let cpu_processor = CpuImageProcessor::new();
    let gpu_processor = GpuImageProcessor::new(&Device::gpu(0)).unwrap();

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
            dest_format: ImageFormat::RGB888,
            fit_mode: FitMode::STRETCH,
            rotation: Rotation::None,
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
            dest_format: ImageFormat::RGBF32,
            fit_mode: FitMode::CONTAIN,
            rotation: Rotation::R90DEG,
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
            dest_format: ImageFormat::RGB888,
            fit_mode: FitMode::CROP,
            rotation: Rotation::R180DEG,
        },
    ];

    for opts in test_cases {
        let cpu_buf = cpu_processor.process(&input, &opts).unwrap();
        let gpu_buf = gpu_processor.process(&input, &opts).unwrap();

        assert_eq!(cpu_buf.shape(), gpu_buf.shape());
        assert_eq!(cpu_buf.dtype(), gpu_buf.dtype());
        assert_eq!(gpu_buf.device(), &Device::gpu(0));

        let cpu_host = cpu_buf.read_to_cpu().unwrap();
        let gpu_host = gpu_buf.read_to_cpu().unwrap();

        if opts.dest_format == ImageFormat::RGB888 {
            let cpu_bytes = cpu_host.as_slice_u8().unwrap();
            let gpu_bytes = gpu_host.as_slice_u8().unwrap();
            assert_eq!(cpu_bytes.len(), gpu_bytes.len());

            // Check difference (small bilinear filter differences are allowed)
            let mut diff_sum: u64 = 0;
            for (c, g) in cpu_bytes.iter().zip(gpu_bytes.iter()) {
                diff_sum += (*c as i64 - *g as i64).unsigned_abs();
            }
            let avg_diff = (diff_sum as f64) / (cpu_bytes.len() as f64);
            assert!(
                avg_diff < 5.0,
                "Average pixel difference {} exceeded tolerance for {:?}",
                avg_diff,
                opts.fit_mode
            );
        } else {
            let cpu_f32 = cpu_host.as_slice_f32().unwrap();
            let gpu_f32 = gpu_host.as_slice_f32().unwrap();
            assert_eq!(cpu_f32.len(), gpu_f32.len());

            let mut diff_sum: f32 = 0.0;
            for (c, g) in cpu_f32.iter().zip(gpu_f32.iter()) {
                diff_sum += (c - g).abs();
            }
            let avg_diff = diff_sum / (cpu_f32.len() as f32);
            assert!(
                avg_diff < 0.02,
                "Average float difference {} exceeded tolerance for {:?}",
                avg_diff,
                opts.fit_mode
            );
        }
    }
}
