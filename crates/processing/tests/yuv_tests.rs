use infers_core::{DataType, HardwareImage, ImageFormat, ProcessingOptions};
use processing_core::{FitMode, TensorLayout};
use processing::{CpuImageProcessor, DeferredCpuProcessExt};

#[cfg(feature = "vulkan")]
use infers_gpu::{defer_hardware, Vulkan};
#[cfg(feature = "vulkan")]
use processing::{DeferredVulkanProcessExt, GpuImageProcessor};

fn rgb_to_nv12(width: u32, height: u32, rgb: &[u8]) -> Vec<u8> {
    let y_size = (width * height) as usize;
    let mut nv12 = vec![0u8; y_size + y_size / 2];
    for y in 0..height {
        for x in 0..width {
            let i = ((y * width + x) * 3) as usize;
            let r = rgb[i] as f32;
            let g = rgb[i + 1] as f32;
            let b = rgb[i + 2] as f32;
            let luma = (0.299 * r + 0.587 * g + 0.114 * b).round().clamp(0.0, 255.0) as u8;
            nv12[(y * width + x) as usize] = luma;
            if y % 2 == 0 && x % 2 == 0 {
                let u = (-0.168736 * r - 0.331264 * g + 0.5 * b + 128.0)
                    .round()
                    .clamp(0.0, 255.0) as u8;
                let v = (0.5 * r - 0.418688 * g - 0.081312 * b + 128.0)
                    .round()
                    .clamp(0.0, 255.0) as u8;
                let uv = y_size + (y / 2 * width + x) as usize;
                nv12[uv] = u;
                nv12[uv + 1] = v;
            }
        }
    }
    nv12
}

fn rgb_to_i420(width: u32, height: u32, rgb: &[u8]) -> Vec<u8> {
    let y_size = (width * height) as usize;
    let cw = (width / 2) as usize;
    let ch = (height / 2) as usize;
    let mut i420 = vec![0u8; y_size + cw * ch * 2];
    for y in 0..height {
        for x in 0..width {
            let i = ((y * width + x) * 3) as usize;
            let r = rgb[i] as f32;
            let g = rgb[i + 1] as f32;
            let b = rgb[i + 2] as f32;
            let luma = (0.299 * r + 0.587 * g + 0.114 * b).round().clamp(0.0, 255.0) as u8;
            i420[(y * width + x) as usize] = luma;
            if y % 2 == 0 && x % 2 == 0 {
                let u = (-0.168736 * r - 0.331264 * g + 0.5 * b + 128.0)
                    .round()
                    .clamp(0.0, 255.0) as u8;
                let v = (0.5 * r - 0.418688 * g - 0.081312 * b + 128.0)
                    .round()
                    .clamp(0.0, 255.0) as u8;
                let cx = (x / 2) as usize;
                let cy = (y / 2) as usize;
                i420[y_size + cy * cw + cx] = u;
                i420[y_size + cw * ch + cy * cw + cx] = v;
            }
        }
    }
    i420
}

#[cfg(feature = "vulkan")]
fn try_gpu_processor() -> Option<(Vulkan, GpuImageProcessor)> {
    let vulkan = Vulkan::new(0).ok()?;
    let processor = GpuImageProcessor::new(vulkan.clone()).ok()?;
    Some((vulkan, processor))
}

#[test]
fn test_cpu_nv12_to_rgb888() {
    let width = 32u32;
    let height = 24u32;
    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            rgb.push((x * 8) as u8);
            rgb.push((y * 10) as u8);
            rgb.push(128);
        }
    }
    let nv12 = rgb_to_nv12(width, height, &rgb);
    let host = HardwareImage::new(width, height, ImageFormat::Nv12, nv12).unwrap();
    let processor = CpuImageProcessor::new();
    let opts = ProcessingOptions {
        src_w: width,
        src_h: height,
        dest_w: width,
        dest_h: height,
        dest_format: ImageFormat::Rgb888,
        fit_mode: FitMode::Stretch,
        ..Default::default()
    };
    let out = host
        .on_cpu()
        .process(&processor, &opts)
        .unwrap()
        .materialize()
        .unwrap();
    assert_eq!(out.dtype(), DataType::U8);
    let host = out.read_to_host().unwrap();
    let bytes = host.as_slice_u8().unwrap();
    let mut err = 0u64;
    for i in 0..bytes.len() {
        err += (bytes[i] as i16 - rgb[i] as i16).unsigned_abs() as u64;
    }
    let avg = err as f64 / bytes.len() as f64;
    assert!(avg < 8.0, "NV12 roundtrip average error {avg}");
}

#[test]
fn test_cpu_i420_to_rgbf32() {
    let width = 16u32;
    let height = 16u32;
    let rgb = [200u8, 40, 40].repeat((width * height) as usize);
    let i420 = rgb_to_i420(width, height, &rgb);
    let host = HardwareImage::new(width, height, ImageFormat::I420, i420).unwrap();
    let processor = CpuImageProcessor::new();
    let opts = ProcessingOptions {
        src_w: width,
        src_h: height,
        dest_w: 8,
        dest_h: 8,
        dest_format: ImageFormat::Rgbf32,
        dest_layout: TensorLayout::default_for_dest_format(ImageFormat::Rgbf32),
        fit_mode: FitMode::Stretch,
        ..Default::default()
    };
    let out = host
        .on_cpu()
        .process(&processor, &opts)
        .unwrap()
        .materialize()
        .unwrap();
    assert_eq!(out.dtype(), DataType::F32);
    let host = out.read_to_host().unwrap();
    let vals = host.as_slice_f32().unwrap();
    let hw = 8 * 8;
    let r_plane = &vals[0..hw];
    let g_plane = &vals[hw..2 * hw];
    let b_plane = &vals[2 * hw..3 * hw];
    let r_sum: f32 = r_plane.iter().sum();
    for (&r, (&g, &b)) in r_plane.iter().zip(g_plane.iter().zip(b_plane.iter())) {
        assert!((0.0..=1.0).contains(&r));
        assert!(r > g && r > b);
    }
    assert!(r_sum / hw as f32 > 0.5);
}

#[test]
#[cfg(feature = "vulkan")]
fn test_gpu_nv12_to_rgb888() {
    let width = 32u32;
    let height = 24u32;
    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            rgb.push((x * 8) as u8);
            rgb.push((y * 10) as u8);
            rgb.push(128);
        }
    }
    let nv12 = rgb_to_nv12(width, height, &rgb);
    let host = HardwareImage::new(width, height, ImageFormat::Nv12, nv12).unwrap();
    let (vulkan, processor) = match try_gpu_processor() {
        Some(p) => p,
        None => {
            eprintln!("skipping GPU NV12 test: Vulkan unavailable");
            return;
        }
    };
    let opts = ProcessingOptions {
        src_w: width,
        src_h: height,
        dest_w: width,
        dest_h: height,
        dest_format: ImageFormat::Rgb888,
        fit_mode: FitMode::Stretch,
        ..Default::default()
    };
    let out = defer_hardware(&vulkan, host)
        .process(&processor, &opts)
        .unwrap()
        .materialize()
        .unwrap();
    assert_eq!(out.dtype(), DataType::U8);
    let host = out.read_to_host().unwrap();
    let bytes = host.as_slice_u8().unwrap();
    let mut err = 0u64;
    for i in 0..bytes.len() {
        err += (bytes[i] as i16 - rgb[i] as i16).unsigned_abs() as u64;
    }
    let avg = err as f64 / bytes.len() as f64;
    assert!(avg < 8.0, "NV12 roundtrip average error {avg}");
}

#[test]
#[cfg(feature = "vulkan")]
fn test_gpu_i420_to_rgbf32() {
    let width = 16u32;
    let height = 16u32;
    let rgb = [200u8, 40, 40].repeat((width * height) as usize);
    let i420 = rgb_to_i420(width, height, &rgb);
    let host = HardwareImage::new(width, height, ImageFormat::I420, i420).unwrap();
    let (vulkan, processor) = match try_gpu_processor() {
        Some(p) => p,
        None => {
            eprintln!("skipping GPU I420 test: Vulkan unavailable");
            return;
        }
    };
    let opts = ProcessingOptions {
        src_w: width,
        src_h: height,
        dest_w: 8,
        dest_h: 8,
        dest_format: ImageFormat::Rgbf32,
        dest_layout: TensorLayout::default_for_dest_format(ImageFormat::Rgbf32),
        fit_mode: FitMode::Stretch,
        ..Default::default()
    };
    let out = defer_hardware(&vulkan, host)
        .process(&processor, &opts)
        .unwrap()
        .materialize()
        .unwrap();
    assert_eq!(out.dtype(), DataType::F32);
    let host = out.read_to_host().unwrap();
    let vals = host.as_slice_f32().unwrap();
    let hw = 8 * 8;
    let r_plane = &vals[0..hw];
    let g_plane = &vals[hw..2 * hw];
    let b_plane = &vals[2 * hw..3 * hw];
    let r_sum: f32 = r_plane.iter().sum();
    for (&r, (&g, &b)) in r_plane.iter().zip(g_plane.iter().zip(b_plane.iter())) {
        assert!((0.0..=1.0).contains(&r));
        assert!(r > g && r > b);
    }
    assert!(r_sum / hw as f32 > 0.5);
}
