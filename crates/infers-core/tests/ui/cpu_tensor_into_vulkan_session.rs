use infers_core::{Cpu, HostTensor, Session, Tensor, TensorShape};
use infers_gpu::Vulkan;
use infers_test_utils::mock_gpu_detector;

fn main() {
    let vulkan = Vulkan::new(0).unwrap();
    let mut session = mock_gpu_detector(vulkan.clone(), 224);
    let host = HostTensor::from_f32(TensorShape::new([1]).unwrap(), vec![0.0]).unwrap();
    let cpu_tensor = Tensor::from_host(&Cpu, &host).unwrap();
    session.infer(&cpu_tensor).unwrap();
}
