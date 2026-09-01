//! Benchmark the GPU inference pipeline (Vulkan preprocess → Vulkan ET).

#[path = "common/mod.rs"]
mod common;

#[cfg(not(feature = "vulkan"))]
fn main() {
    eprintln!(
        "gpu_pipeline_bench requires the `vulkan` feature.\n\
         Try: cargo run --release --example gpu_pipeline_bench --features vulkan"
    );
    std::process::exit(1);
}

#[cfg(feature = "vulkan")]
fn main() {
    use common::{
        camera_frame, detector_options, imgsz_from_input_shape, require_model, BenchArgs,
        StageStats,
    };
    use infers::{
        Device, ExecuTorchBackend, GpuImageProcessor, Session, Vulkan, VulkanOptions,
    };

    let (args, model) = BenchArgs::parse("vulkan/model.pte");
    let model_bytes = require_model(&model);

    let vulkan = Vulkan::new(0).unwrap_or_else(|err| {
        panic!("Vulkan::new failed: {err}\nA GPU with Vulkan is required.");
    });

    let processor = GpuImageProcessor::new(vulkan.clone()).expect("GpuImageProcessor::new");

    let backend = ExecuTorchBackend::new();
    let mut session = backend
        .load_vulkan(
            &model_bytes,
            &vulkan,
            VulkanOptions { method: None },
        )
        .expect("load Vulkan yolo26n-face");

    let imgsz = imgsz_from_input_shape(&session.input_shapes()[0]);
    let frame = camera_frame(args.width, args.height);
    let options = detector_options(args.width, args.height, imgsz);
    let gpu_image = vulkan.upload_image(&frame).expect("upload frame");

    println!(
        "GPU pipeline bench\n  model:     {}\n  frame:     {}x{} Rgb888 (resident)\n  input:     1x3x{imgsz}x{imgsz} NCHW\n  warmup:    {}\n  iters:     {}\n",
        model.display(),
        args.width,
        args.height,
        args.warmup,
        args.iters,
    );

    for _ in 0..args.warmup {
        run_once(&processor, &mut session, &gpu_image, &options);
    }

    let mut stats = StageStats::default();
    for _ in 0..args.iters {
        let (preprocess_ms, infer_ms, total_ms) =
            run_once(&processor, &mut session, &gpu_image, &options);
        stats.record(preprocess_ms, infer_ms, total_ms);
    }

    stats.print("GPU (Vulkan preprocess + Vulkan ExecuTorch)");
    let last = run_once_outputs(&processor, &mut session, &gpu_image, &options);
    println!(
        "  last run outputs: {} tensor(s), first shape {:?}",
        last.len(),
        last[0].shape().dims()
    );
}

#[cfg(feature = "vulkan")]
use common::timed;

#[cfg(feature = "vulkan")]
use infers::Session;

#[cfg(feature = "vulkan")]
fn run_once(
    processor: &infers::GpuImageProcessor,
    session: &mut infers::ExecuTorchSession<infers::Vulkan>,
    image: &infers_gpu::VulkanImage,
    options: &infers::ProcessingOptions,
) -> (std::time::Duration, std::time::Duration, std::time::Duration) {
    let total_start = std::time::Instant::now();

    let (preprocessed, preprocess) = timed(|| {
        processor.process(image, options).expect("GPU preprocess")
    });

    let (_outputs, infer) = timed(|| {
        session.run(&[&preprocessed]).expect("Vulkan inference")
    });

    (preprocess, infer, total_start.elapsed())
}

#[cfg(feature = "vulkan")]
fn run_once_outputs(
    processor: &infers::GpuImageProcessor,
    session: &mut infers::ExecuTorchSession<infers::Vulkan>,
    image: &infers_gpu::VulkanImage,
    options: &infers::ProcessingOptions,
) -> Vec<infers::Tensor<infers::Cpu>> {
    let preprocessed = processor.process(image, options).expect("GPU preprocess");
    session.run(&[&preprocessed]).expect("Vulkan inference")
}
