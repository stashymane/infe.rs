//! Benchmark the CPU inference pipeline (preprocess → XNNPACK).
//!
//! The camera frame is allocated once up front and excluded from timings, matching
//! a host that already holds a buffer from the device camera.
//!
//! ```text
//! cargo run --release --example cpu_pipeline_bench
//! cargo run --release --example cpu_pipeline_bench -- --iters 100 --threads 4
//! ```

#[path = "common/mod.rs"]
mod common;

use common::{camera_frame, detector_options, imgsz_from_input_shape, require_model, timed, BenchArgs, StageStats};
use infers::{
    CpuImageProcessor, ExecuTorchBackend, ExecuTorchBackendConfig, ModelSession, TensorBuffer,
};
use std::time::Duration;

fn main() {
    let (args, model) = BenchArgs::parse("xnnpack/model.pte");
    let model_bytes = require_model(&model);

    let backend = ExecuTorchBackend::new();
    let mut session = backend
        .load_model(
            &model_bytes,
            ExecuTorchBackendConfig::Xnnpack {
                num_threads: args.threads,
                method: None,
            },
        )
        .expect("load XNNPACK yolo26n-face");

    let imgsz = imgsz_from_input_shape(&session.input_shapes()[0]);
    let frame = camera_frame(args.width, args.height);
    let options = detector_options(args.width, args.height, imgsz);
    let processor = CpuImageProcessor::new();

    println!(
        "CPU pipeline bench\n  model:     {}\n  frame:     {}x{} Rgb888 (resident)\n  input:     1x3x{imgsz}x{imgsz} NCHW\n  threads:   {}\n  warmup:    {}\n  iters:     {}\n",
        model.display(),
        args.width,
        args.height,
        args.threads,
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

    stats.print("CPU (XNNPACK)");
    let last = run_once_outputs(&processor, &mut *session, &frame, &options);
    println!(
        "  last run outputs: {} tensor(s), first shape {:?}",
        last.len(),
        last[0].shape().dims()
    );
}

fn run_once(
    processor: &CpuImageProcessor,
    session: &mut dyn ModelSession,
    frame: &infers::CpuImageBuffer,
    options: &infers::ProcessingOptions,
) -> (Duration, Duration, Duration) {
    let total_start = std::time::Instant::now();

    let (preprocessed, preprocess) = timed(|| {
        processor
            .process(frame, options)
            .expect("CPU preprocess")
    });

    let (_outputs, infer) = timed(|| {
        let input: &dyn TensorBuffer = preprocessed.as_ref();
        session.run(&[input]).expect("XNNPACK inference")
    });

    (preprocess, infer, total_start.elapsed())
}

fn run_once_outputs(
    processor: &CpuImageProcessor,
    session: &mut dyn ModelSession,
    frame: &infers::CpuImageBuffer,
    options: &infers::ProcessingOptions,
) -> Vec<Box<dyn TensorBuffer>> {
    let preprocessed = processor.process(frame, options).expect("CPU preprocess");
    session
        .run(&[preprocessed.as_ref()])
        .expect("XNNPACK inference")
}