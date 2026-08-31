//! Benchmark the GPU inference pipeline (Vulkan preprocess → Vulkan ET).
//!
//! The camera frame is allocated once up front and excluded from timings, matching
//! a host that already holds a buffer from the device camera (or an imported
//! `AHardwareBuffer` on Android).
//!
//! ```text
//! cargo run --release --example gpu_pipeline_bench
//! cargo run --release --example gpu_pipeline_bench -- --iters 100
//! ```
//!
//! Preprocess emits NCHW directly on the GPU and feeds ExecuTorch Vulkan staging
//! via a GPU→GPU copy in `prepare_inputs`.

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
    use std::sync::Arc;

    use common::{
        camera_frame, detector_options, imgsz_from_input_shape, require_model, BenchArgs,
        StageStats,
    };
    use infers::{
        Device, ExecuTorchBackend, ExecuTorchBackendConfig, GpuImageProcessor, VulkanContext,
    };

    let (args, model) = BenchArgs::parse("vulkan/model.pte");
    let model_bytes = require_model(&model);

    let ctx = Arc::new(
        VulkanContext::new(&Device::gpu(0)).unwrap_or_else(|err| {
            panic!("VulkanContext::new failed: {err}\nA GPU with Vulkan is required.");
        }),
    );

    let processor =
        GpuImageProcessor::new(Arc::clone(&ctx)).expect("GpuImageProcessor::new");

    let backend = ExecuTorchBackend::new();
    let mut session = backend
        .load_model(
            &model_bytes,
            ExecuTorchBackendConfig::Vulkan {
                context: Arc::clone(&ctx),
                method: None,
            },
        )
        .expect("load Vulkan yolo26n-face");

    let imgsz = imgsz_from_input_shape(&session.input_shapes()[0]);
    let frame = camera_frame(args.width, args.height);
    let options = detector_options(args.width, args.height, imgsz);

    println!(
        "GPU pipeline bench\n  model:     {}\n  frame:     {}x{} Rgb888 (resident)\n  input:     1x3x{imgsz}x{imgsz} NCHW\n  warmup:    {}\n  iters:     {}\n",
        model.display(),
        args.width,
        args.height,
        args.warmup,
        args.iters,
    );

    for _ in 0..args.warmup {
        run_once(&processor, &mut *session, &frame, &options);
    }

    let mut stats = StageStats::default();
    for _ in 0..args.iters {
        let (preprocess_ms, infer_ms, total_ms) =
            run_once(&processor, &mut *session, &frame, &options);
        stats.record(preprocess_ms, infer_ms, total_ms);
    }

    stats.print("GPU (Vulkan preprocess + Vulkan ExecuTorch)");
    let last = run_once_outputs(&processor, &mut *session, &frame, &options);
    println!(
        "  last run outputs: {} tensor(s), first shape {:?}",
        last.len(),
        last[0].shape().dims()
    );
}

#[cfg(feature = "vulkan")]
fn run_once(
    processor: &infers::GpuImageProcessor,
    session: &mut dyn infers::ModelSession,
    frame: &infers::CpuImageBuffer,
    options: &infers::ProcessingOptions,
) -> (
    std::time::Duration,
    std::time::Duration,
    std::time::Duration,
) {
    use common::timed;
    use infers::TensorBuffer;

    let total_start = std::time::Instant::now();

    let (preprocessed, preprocess) = timed(|| {
        processor
            .process(frame, options)
            .expect("GPU preprocess")
    });

    let (_outputs, infer) = timed(|| {
        let input: &dyn TensorBuffer = preprocessed.as_ref();
        session.run(&[input]).expect("Vulkan inference")
    });

    (preprocess, infer, total_start.elapsed())
}

#[cfg(feature = "vulkan")]
fn run_once_outputs(
    processor: &infers::GpuImageProcessor,
    session: &mut dyn infers::ModelSession,
    frame: &infers::CpuImageBuffer,
    options: &infers::ProcessingOptions,
) -> Vec<Box<dyn infers::TensorBuffer>> {
    let preprocessed = processor.process(frame, options).expect("GPU preprocess");
    session
        .run(&[preprocessed.as_ref()])
        .expect("Vulkan inference")
}
