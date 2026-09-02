use infers::{
    Cpu, ExecuTorchBackend, ExecuTorchSession, GpuImageProcessor, ProcessingOptions, Session,
    Vulkan, VulkanImage, VulkanOptions,
};
use processing::ImageProcessor;

use super::fixtures::{
    camera_frame, detector_options, imgsz_from_input_shape, model_path, require_model, FRAME_H,
    FRAME_W,
};

pub struct Bench {
    pub processor: GpuImageProcessor,
    pub session: ExecuTorchSession<Vulkan>,
    pub image: VulkanImage,
    pub options: ProcessingOptions,
}

pub fn setup() -> Bench {
    let model_bytes = require_model(&model_path("vulkan/model.pte"));
    let vulkan = Vulkan::new(0).expect("Vulkan::new requires a GPU with Vulkan support");
    let processor = GpuImageProcessor::new(vulkan.clone()).expect("GpuImageProcessor::new");

    let backend = ExecuTorchBackend::new();
    let session = backend
        .load_vulkan(
            &model_bytes,
            &vulkan,
            VulkanOptions { method: None },
        )
        .expect("load Vulkan yolo26n-face");

    let imgsz = imgsz_from_input_shape(&session.input_shapes()[0]);
    let frame = camera_frame(FRAME_W, FRAME_H);
    let options = detector_options(FRAME_W, FRAME_H, imgsz);

    let mut buffer = vulkan
        .image_buffer(frame.width(), frame.height(), frame.format())
        .expect("allocate GPU image buffer");
    buffer.write(frame.as_bytes()).expect("upload frame");
    let image = VulkanImage::Linear(buffer);

    Bench {
        processor,
        session,
        image,
        options,
    }
}

pub fn preprocess(bench: &Bench) -> infers::Pending<Vulkan> {
    bench
        .processor
        .process(&bench.image, &bench.options)
        .expect("GPU preprocess")
}

pub fn infer(
    bench: &mut Bench,
    input: impl infers::InferInput<Vulkan>,
) -> Vec<infers::Tensor<Cpu>> {
    bench.session.infer(input).expect("Vulkan inference")
}
