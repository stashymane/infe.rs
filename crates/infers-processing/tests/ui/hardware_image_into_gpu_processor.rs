use infers_core::{HardwareImage, ImageFormat};
use infers_gpu::Vulkan;
use infers_processing::GpuImageProcessor;

fn main() {
    let vulkan = Vulkan::new(0).unwrap();
    let proc = GpuImageProcessor::new(vulkan).unwrap();
    let host = HardwareImage::new(4, 4, ImageFormat::Rgb888, vec![0; 48]).unwrap();
    let _ = proc.process(&host, &infers_processing_core::ProcessingOptions::default());
}
