use infers::{
    Cpu, CpuImageProcessor, ExecuTorchBackend, ExecuTorchSession, Session, Tensor, XnnpackOptions,
};

use super::fixtures::{
    camera_frame, detector_options, imgsz_from_input_shape, model_path, require_model, FRAME_H,
    FRAME_W,
};

const XNNPACK_THREADS: usize = 4;

pub struct Bench {
    pub processor: CpuImageProcessor,
    pub session: ExecuTorchSession<Cpu>,
    pub frame: infers::HostImage,
    pub options: infers::ProcessingOptions,
}

pub fn setup() -> Bench {
    let model_bytes = require_model(&model_path("xnnpack/model.pte"));
    let backend = ExecuTorchBackend::new();
    let session = backend
        .load_xnnpack(
            &model_bytes,
            XnnpackOptions {
                num_threads: XNNPACK_THREADS,
                method: None,
            },
        )
        .expect("load XNNPACK yolo26n-face");

    let imgsz = imgsz_from_input_shape(&session.input_shapes()[0]);
    let frame = camera_frame(FRAME_W, FRAME_H);
    let options = detector_options(FRAME_W, FRAME_H, imgsz);

    Bench {
        processor: CpuImageProcessor::new(),
        session,
        frame,
        options,
    }
}

pub fn preprocess(bench: &Bench) -> Tensor<Cpu> {
    bench
        .processor
        .process(&bench.frame, &bench.options)
        .expect("CPU preprocess")
}

pub fn infer(bench: &mut Bench, input: &Tensor<Cpu>) -> Vec<Tensor<Cpu>> {
    bench.session.run(&[input]).expect("XNNPACK inference")
}
