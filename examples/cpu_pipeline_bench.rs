//! Benchmark the CPU inference pipeline (preprocess → XNNPACK).

#[path = "common/mod.rs"]
mod common;

use common::{camera_frame, detector_options, imgsz_from_input_shape, require_model, timed, BenchArgs, StageStats};
use infers::{
    CpuImageProcessor, ExecuTorchBackend, Session, XnnpackOptions,
};

fn main() {
    let (args, model) = BenchArgs::parse("xnnpack/model.pte");
    let model_bytes = require_model(&model);

    let backend = ExecuTorchBackend::new();
    let mut session = backend
        .load_xnnpack(
            &model_bytes,
            XnnpackOptions {
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
        run_once(&processor, &mut session, &frame, &options);
    }

    let mut stats = StageStats::default();
    for _ in 0..args.iters {
        let (preprocess_ms, infer_ms, total_ms) =
            run_once(&processor, &mut session, &frame, &options);
        stats.record(preprocess_ms, infer_ms, total_ms);
    }

    stats.print("CPU (XNNPACK)");
    let last = run_once_outputs(&processor, &mut session, &frame, &options);
    println!(
        "  last run outputs: {} tensor(s), first shape {:?}",
        last.len(),
        last[0].shape().dims()
    );
}

fn run_once(
    processor: &CpuImageProcessor,
    session: &mut infers::ExecuTorchSession<infers::Cpu>,
    frame: &infers::HostImage,
    options: &infers::ProcessingOptions,
) -> (std::time::Duration, std::time::Duration, std::time::Duration) {
    let total_start = std::time::Instant::now();

    let (preprocessed, preprocess) = timed(|| {
        processor.process(frame, options).expect("CPU preprocess")
    });

    let (_outputs, infer) = timed(|| {
        session.run(&[&preprocessed]).expect("XNNPACK inference")
    });

    (preprocess, infer, total_start.elapsed())
}

fn run_once_outputs(
    processor: &CpuImageProcessor,
    session: &mut infers::ExecuTorchSession<infers::Cpu>,
    frame: &infers::HostImage,
    options: &infers::ProcessingOptions,
) -> Vec<infers::Tensor<infers::Cpu>> {
    let preprocessed = processor.process(frame, options).expect("CPU preprocess");
    session.run(&[&preprocessed]).expect("XNNPACK inference")
}
