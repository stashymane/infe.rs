use infers_core::{CpuImageBuffer, DataType, Device, ImageFormat, ProcessingOptions};
use infers_gpu::VulkanContext;
use processing_core::FitMode;
use processing_gpu::GpuImageProcessor;
use std::sync::Arc;

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

fn try_gpu_processor() -> Option<GpuImageProcessor> {
    let device = Device::gpu(0);
    let context = Arc::new(VulkanContext::new(&device).ok()?);
    GpuImageProcessor::new(context).ok()
}

#[test]
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
    let input = CpuImageBuffer::new(width, height, ImageFormat::NV12, nv12).unwrap();
    let processor = match try_gpu_processor() {
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
        dest_format: ImageFormat::RGB888,
        fit_mode: FitMode::STRETCH,
        ..Default::default()
    };
    let out = processor.process(&input, &opts).unwrap();
    assert_eq!(out.dtype(), DataType::U8);
    let host = out.read_to_cpu().unwrap();
    let bytes = host.as_slice_u8().unwrap();
    let mut err = 0u64;
    for i in 0..bytes.len() {
        err += (bytes[i] as i16 - rgb[i] as i16).unsigned_abs() as u64;
    }
    let avg = err as f64 / bytes.len() as f64;
    assert!(avg < 8.0, "NV12 roundtrip average error {avg}");
}

#[test]
fn test_gpu_i420_to_rgbf32() {
    let width = 16u32;
    let height = 16u32;
    let rgb = vec![200u8, 40, 40].repeat((width * height) as usize);
    let i420 = rgb_to_i420(width, height, &rgb);
    let input = CpuImageBuffer::new(width, height, ImageFormat::I420, i420).unwrap();
    let processor = match try_gpu_processor() {
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
        dest_format: ImageFormat::RGBF32,
        fit_mode: FitMode::STRETCH,
        ..Default::default()
    };
    let out = processor.process(&input, &opts).unwrap();
    assert_eq!(out.dtype(), DataType::F32);
    let host = out.read_to_cpu().unwrap();
    let vals = host.as_slice_f32().unwrap();
    let mut r_sum = 0.0;
    for chunk in vals.chunks_exact(3) {
        r_sum += chunk[0];
        assert!((0.0..=1.0).contains(&chunk[0]));
        assert!(chunk[0] > chunk[1] && chunk[0] > chunk[2]);
    }
    assert!(r_sum / (vals.len() / 3) as f32 > 0.5);
}
