use infers::{
    Cpu, Device, ExecuTorchBackend, ExecuTorchSession, GpuImageProcessor, Session, Tensor, Vulkan,
    VulkanOptions,
};
use infers_gpu::VulkanImage;

use super::fixtures::{
    camera_frame, detector_options, imgsz_from_input_shape, model_path, require_model, FRAME_H,
    FRAME_W,
};

pub struct Bench {
    pub processor: GpuImageProcessor,
    pub session: ExecuTorchSession<Vulkan>,
    pub image: VulkanImage,
    pub options: infers::ProcessingOptions,
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
    let image = vulkan.upload_image(&frame).expect("upload frame");

    Bench {
        processor,
        session,
        image,
        options,
    }
}

pub fn preprocess(bench: &Bench) -> Tensor<Vulkan> {
    bench
        .processor
        .process(&bench.image, &bench.options)
        .expect("GPU preprocess")
}

pub fn infer(bench: &mut Bench, input: &Tensor<Vulkan>) -> Vec<Tensor<Cpu>> {
    bench.session.run(&[input]).expect("Vulkan inference")
}
