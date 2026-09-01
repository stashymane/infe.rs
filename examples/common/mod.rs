//! Shared helpers for pipeline benchmark examples.
//!
//! The timed path starts after a camera-like frame buffer already exists in memory.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use clap::Parser;
use infers::{HostImage, ImageFormat, ProcessingOptions, Rotation, TensorShape, TensorLayout};
use processing::FitMode;

pub const DEFAULT_FRAME_W: u32 = 1280;
pub const DEFAULT_FRAME_H: u32 = 720;
pub const DEFAULT_ITERS: usize = 50;
pub const DEFAULT_WARMUP: usize = 5;
pub const DEFAULT_THREADS: usize = 4;

/// Pipeline benchmark options.
///
/// Timing excludes allocating the camera frame buffer (already resident, as from a
/// device camera). It includes preprocess and inference.
#[derive(Clone, Debug, Parser)]
#[command(
    about = "Benchmark preprocess → ExecuTorch inference",
    after_help = "The camera-like frame is allocated once and excluded from timings."
)]
pub struct BenchArgs {
    /// Path to the .pte model (default: target/yolo26n-face/<backend>/model.pte)
    #[arg(long, value_name = "PATH")]
    pub model: Option<PathBuf>,

    /// Timed iterations
    #[arg(long, default_value_t = DEFAULT_ITERS)]
    pub iters: usize,

    /// Warmup iterations excluded from stats
    #[arg(long, default_value_t = DEFAULT_WARMUP)]
    pub warmup: usize,

    /// Synthetic camera frame width
    #[arg(long, default_value_t = DEFAULT_FRAME_W)]
    pub width: u32,

    /// Synthetic camera frame height
    #[arg(long, default_value_t = DEFAULT_FRAME_H)]
    pub height: u32,

    /// XNNPACK thread count (CPU example only)
    #[arg(long, default_value_t = DEFAULT_THREADS)]
    pub threads: usize,
}

impl BenchArgs {
    pub fn parse(default_model_rel: &str) -> (Self, PathBuf) {
        let args = Self::parse_from(std::env::args_os());
        let model = args
            .model
            .clone()
            .unwrap_or_else(|| default_asset(default_model_rel));
        (args, model)
    }
}

pub fn default_asset(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/yolo26n-face")
        .join(rel)
}

pub fn require_model(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|err| {
        panic!(
            "failed to read model '{}': {err}\n\
             Build models first (see assets/yolo26n-face-manifest.yaml, output in target/yolo26n-face/).",
            path.display()
        )
    })
}

/// Synthetic camera frame already in memory — not part of the timed path.
pub fn camera_frame(width: u32, height: u32) -> HostImage {
    let mut data = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            data.push(((x * 255) / width.max(1)) as u8);
            data.push(((y * 255) / height.max(1)) as u8);
            data.push((((x + y) * 255) / (width + height).max(1)) as u8);
        }
    }
    HostImage::new(width, height, ImageFormat::Rgb888, data).expect("synthetic camera frame")
}

pub fn detector_options(src_w: u32, src_h: u32, imgsz: u32) -> ProcessingOptions {
    let dest_format = ImageFormat::Rgbf32;
    ProcessingOptions {
        src_w,
        src_h,
        crop_x: 0,
        crop_y: 0,
        crop_w: src_w,
        crop_h: src_h,
        dest_w: imgsz,
        dest_h: imgsz,
        src_format: ImageFormat::Rgb888,
        dest_format,
        fit_mode: FitMode::Contain,
        rotation: Rotation::None,
        dest_layout: TensorLayout::default_for_dest_format(dest_format),
    }
}

pub fn imgsz_from_input_shape(shape: &TensorShape) -> u32 {
    let dims = shape.dims();
    assert_eq!(
        dims.len(),
        4,
        "expected NCHW input shape [1, 3, H, W], got {dims:?}"
    );
    assert_eq!(dims[0], 1, "batch must be 1");
    assert_eq!(dims[1], 3, "channels must be 3 (NCHW)");
    assert_eq!(dims[2], dims[3], "square model input expected");
    dims[2] as u32
}

#[derive(Default, Clone)]
pub struct StageStats {
    pub preprocess: Vec<Duration>,
    pub infer: Vec<Duration>,
    pub total: Vec<Duration>,
}

impl StageStats {
    pub fn record(&mut self, preprocess: Duration, infer: Duration, total: Duration) {
        self.preprocess.push(preprocess);
        self.infer.push(infer);
        self.total.push(total);
    }

    pub fn print(&self, label: &str) {
        println!("=== {label} ===");
        print_stage("preprocess", &self.preprocess);
        print_stage("inference", &self.infer);
        print_stage("total (hot path)", &self.total);
    }
}

fn print_stage(name: &str, samples: &[Duration]) {
    let (mean, median, min, max) = summarize(samples);
    println!(
        "  {name:<22} mean {:>8.3} ms | median {:>8.3} ms | min {:>8.3} ms | max {:>8.3} ms",
        mean.as_secs_f64() * 1e3,
        median.as_secs_f64() * 1e3,
        min.as_secs_f64() * 1e3,
        max.as_secs_f64() * 1e3,
    );
}

fn summarize(samples: &[Duration]) -> (Duration, Duration, Duration, Duration) {
    assert!(!samples.is_empty(), "no samples");
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let sum: Duration = samples.iter().copied().sum();
    let mean = sum / samples.len() as u32;
    let median = sorted[sorted.len() / 2];
    (
        mean,
        median,
        *sorted.first().unwrap(),
        *sorted.last().unwrap(),
    )
}

pub fn timed<T>(f: impl FnOnce() -> T) -> (T, Duration) {
    let start = Instant::now();
    let value = f();
    (value, start.elapsed())
}
