use infers::{
    Cpu, CpuImage, CpuImageProcessor, ExecuTorchBackend, ExecuTorchSession, ProcessingOptions,
    Session, XnnpackOptions,
};
use infers_processing::ImageProcessor;

use super::fixtures::{FRAME_H, FRAME_W,
                      camera_frame, detector_options, imgsz_from_input_shape,
};

const XNNPACK_MODEL: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../target/yolo26n-face/xnnpack/model.pte"
));

const XNNPACK_THREADS: usize = 2;

pub struct Bench {
    pub processor: CpuImageProcessor,
    pub session: ExecuTorchSession<Cpu>,
    pub image: CpuImage,
    pub options: ProcessingOptions,
}

pub fn setup() -> Bench {
    let backend = ExecuTorchBackend::new();
    let session = backend
        .load_xnnpack(
            XNNPACK_MODEL,
            XnnpackOptions {
                num_threads: XNNPACK_THREADS,
                method: None,
            },
        )
        .expect("load XNNPACK yolo26n-face");

    let imgsz = imgsz_from_input_shape(&session.input_shapes()[0]);
    let frame = camera_frame(FRAME_W, FRAME_H);
    let options = detector_options(FRAME_W, FRAME_H, imgsz);
    let image = frame.on_cpu().materialize().expect("materialize frame");

    Bench {
        processor: CpuImageProcessor::new(),
        session,
        image,
        options,
    }
}

pub fn preprocess(bench: &Bench) -> infers::Pending<Cpu> {
    bench
        .processor
        .process(&bench.image, &bench.options)
        .expect("CPU preprocess")
}

pub fn infer(
    bench: &mut Bench,
    input: impl infers::InferInput<Cpu>,
) -> Vec<infers::Tensor<Cpu>> {
    bench.session.infer(input).expect("XNNPACK inference")
}
