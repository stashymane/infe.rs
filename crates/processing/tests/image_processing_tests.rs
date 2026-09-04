use infers_core::{Cpu, DataType, HardwareImage, ImageFormat, ProcessingOptions, Rotation, Tensor};
use processing_core::TensorLayout;
#[cfg(feature = "vulkan")]
use infers_gpu::{defer_hardware, Vulkan};
use processing::{CpuImageProcessor, DeferredCpuProcessExt, FitMode};
#[cfg(feature = "vulkan")]
use processing::{DeferredVulkanProcessExt, GpuImageProcessor};

fn process_cpu(
    processor: &CpuImageProcessor,
    input: &HardwareImage,
    opts: &ProcessingOptions,
) -> Tensor<Cpu> {
    input
        .clone()
        .on_cpu()
        .process(processor, opts)
        .unwrap()
        .materialize()
        .unwrap()
}

#[cfg(feature = "vulkan")]
fn process_gpu(
    processor: &GpuImageProcessor,
    vulkan: &Vulkan,
    input: &HardwareImage,
    opts: &ProcessingOptions,
) -> Tensor<Vulkan> {
    defer_hardware(vulkan, input.clone())
        .process(processor, opts)
        .unwrap()
        .materialize()
        .unwrap()
}

#[cfg(feature = "vulkan")]
fn materialize_gpu(vulkan: &Vulkan, input: &HardwareImage) -> infers_gpu::VulkanImage {
    defer_hardware(vulkan, input.clone())
        .materialize()
        .unwrap()
}

fn create_test_pattern_image(width: u32, height: u32) -> (Vec<u8>, HardwareImage) {
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
    let buf = HardwareImage::new(width, height, ImageFormat::Rgb888, data.clone()).unwrap();
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

    let tensor_buf = process_cpu(&processor, &input, &opts);
    assert_eq!(tensor_buf.shape().dims(), &[1, 32, 32, 3]);
    assert_eq!(tensor_buf.dtype(), DataType::U8);
    assert_eq!(tensor_buf.device(), &Cpu);

    let host = tensor_buf.read_to_host().unwrap();
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

    let tensor_buf = process_cpu(&processor, &input, &opts);
    assert_eq!(tensor_buf.shape().dims(), &[1, 3, 64, 64]);
    assert_eq!(tensor_buf.dtype(), DataType::F32);

    let host = tensor_buf.read_to_host().unwrap();
    let f32_slice = host.as_slice_f32().unwrap();
    assert_eq!(f32_slice.len(), 64 * 64 * 3);

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

        let tensor_buf = process_cpu(&processor, &input, &opts);
        assert_eq!(tensor_buf.shape().dims(), &[1, 48, 48, 3]);
        let host = tensor_buf.read_to_host().unwrap();
        assert_eq!(host.as_slice_u8().unwrap().len(), 48 * 48 * 3);
    }
}

#[test]
#[cfg(feature = "vulkan")]
fn test_gpu_process_outputs() {
    let (_, host_input) = create_test_pattern_image(80, 60);
    let vulkan = match Vulkan::new(0) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("skipping GPU process test: {err}");
            return;
        }
    };
    let gpu_image = materialize_gpu(&vulkan, &host_input);
    let gpu_processor = match GpuImageProcessor::new(vulkan.clone()) {
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
        let gpu_buf = gpu_processor
            .process(&gpu_image, &opts)
            .unwrap()
            .materialize()
            .unwrap();
        let expected_dims = match opts.dest_format {
            ImageFormat::Rgbf32 => &[1, 3, 32, 32][..],
            _ => &[1, 32, 32, 3],
        };
        assert_eq!(gpu_buf.shape().dims(), expected_dims);

        let gpu_host = gpu_buf.read_to_host().unwrap();
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

#[test]
#[cfg(feature = "vulkan")]
fn test_gpu_process_pooled_repeated_calls_match_cpu() {
    let (_, host_input) = create_test_pattern_image(64, 64);
    let vulkan = match Vulkan::new(0) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("skipping: {err}");
            return;
        }
    };
    let gpu_image = materialize_gpu(&vulkan, &host_input);
    let gpu_processor = GpuImageProcessor::new(vulkan.clone()).expect("processor");

    let opts = ProcessingOptions {
        src_w: 64,
        src_h: 64,
        dest_w: 32,
        dest_h: 32,
        dest_format: ImageFormat::Rgbf32,
        dest_layout: TensorLayout::default_for_dest_format(ImageFormat::Rgbf32),
        fit_mode: FitMode::Stretch,
        rotation: Rotation::None,
        ..Default::default()
    };

    let reference = gpu_processor
        .process(&gpu_image, &opts)
        .unwrap()
        .materialize()
        .unwrap();
    let reference_bytes = reference.read_to_host().unwrap().as_bytes().to_vec();

    for _ in 0..8 {
        let gpu_out = gpu_processor
            .process(&gpu_image, &opts)
            .unwrap()
            .materialize()
            .unwrap();
        let gpu_bytes = gpu_out.read_to_host().unwrap().as_bytes().to_vec();
        assert_eq!(
            gpu_bytes, reference_bytes,
            "pooled GPU output should be deterministic across calls"
        );
    }
}

#[test]
fn test_cpu_identity_memcpy_stretch() {
    let (data, input) = create_test_pattern_image(48, 32);
    let processor = CpuImageProcessor::new();
    let opts = ProcessingOptions {
        src_w: 48,
        src_h: 32,
        dest_w: 48,
        dest_h: 32,
        dest_format: ImageFormat::Rgb888,
        fit_mode: FitMode::Stretch,
        rotation: Rotation::None,
        ..Default::default()
    };
    let out = process_cpu(&processor, &input, &opts);
    let host = out.read_to_host().unwrap();
    assert_eq!(host.as_slice_u8().unwrap(), data.as_slice());
}

#[test]
fn test_cpu_contain_letterbox_zeros() {
    let (_, input) = create_test_pattern_image(100, 50);
    let processor = CpuImageProcessor::new();
    let opts = ProcessingOptions {
        src_w: 100,
        src_h: 50,
        dest_w: 64,
        dest_h: 64,
        dest_format: ImageFormat::Rgb888,
        fit_mode: FitMode::Contain,
        rotation: Rotation::None,
        ..Default::default()
    };
    let out = process_cpu(&processor, &input, &opts);
    let host = out.read_to_host().unwrap();
    let bytes = host.as_slice_u8().unwrap();
    // Top letterbox row should be black.
    assert!(bytes[..64 * 3].iter().all(|&b| b == 0));
    assert!(bytes.iter().any(|&b| b > 0));
}

#[test]
#[cfg(feature = "vulkan")]
fn test_cpu_gpu_parity_storage_path() {
    let (_, host_input) = create_test_pattern_image(80, 60);
    let vulkan = match Vulkan::new(0) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("skipping CPU/GPU parity: {err}");
            return;
        }
    };
    let gpu_processor = match GpuImageProcessor::new(vulkan.clone()) {
        Ok(p) => p,
        Err(err) => {
            eprintln!("skipping CPU/GPU parity: {err}");
            return;
        }
    };
    let cpu_processor = CpuImageProcessor::new();

    let cases = [
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
            dest_w: 32,
            dest_h: 32,
            dest_format: ImageFormat::Rgb888,
            fit_mode: FitMode::Crop,
            rotation: Rotation::Rot180,
            ..Default::default()
        },
    ];

    for opts in cases {
        let cpu_bytes = process_cpu(&cpu_processor, &host_input, &opts)
            .read_to_host()
            .unwrap()
            .as_bytes()
            .to_vec();
        let gpu_bytes = process_gpu(&gpu_processor, &vulkan, &host_input, &opts)
            .read_to_host()
            .unwrap()
            .as_bytes()
            .to_vec();
        assert_eq!(
            cpu_bytes.len(),
            gpu_bytes.len(),
            "byte length mismatch for {opts:?}"
        );
        // Shared convert math can still differ by 1 ULP of rounding between host f32
        // and SPIR-V f32 (bilinear + round-to-u8), especially with Crop + rotation.
        if opts.dest_format == ImageFormat::Rgb888 {
            let mut max_diff = 0u8;
            let mut mismatches = 0usize;
            for (c, g) in cpu_bytes.iter().zip(gpu_bytes.iter()) {
                let d = c.abs_diff(*g);
                max_diff = max_diff.max(d);
                if d > 0 {
                    mismatches += 1;
                }
            }
            assert!(
                max_diff <= 1,
                "RGB888 CPU/GPU max channel diff {max_diff} (mismatched {mismatches}/{}) for {opts:?}",
                cpu_bytes.len()
            );
        } else {
            let cpu_f = bytemuck_f32(&cpu_bytes);
            let gpu_f = bytemuck_f32(&gpu_bytes);
            for (i, (c, g)) in cpu_f.iter().zip(gpu_f.iter()).enumerate() {
                let diff = (c - g).abs();
                assert!(
                    diff <= 1e-4,
                    "f32 mismatch at {i}: cpu={c} gpu={g} diff={diff} opts={opts:?}"
                );
            }
        }
    }
}

fn bytemuck_f32(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len().is_multiple_of(4));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
