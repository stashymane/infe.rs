use infers_core::{
    CoreError, Cpu, CpuImage, Deferred, Device, Pending, ProcessingOptions,
};

#[cfg(feature = "vulkan")]
use infers_gpu::{Vulkan, VulkanImage};

/// Preprocesses materialized device images.
pub trait ImageProcessor<D: Device> {
    fn process(
        &self,
        input: &D::Image,
        options: &ProcessingOptions,
    ) -> Result<Pending<D>, CoreError>;
}

/// Chain preprocess from [`Deferred<Cpu>`].
pub trait DeferredCpuProcessExt {
    fn process(
        self,
        processor: &CpuImageProcessor,
        options: &ProcessingOptions,
    ) -> Result<Pending<Cpu>, CoreError>;
}

impl DeferredCpuProcessExt for Deferred<Cpu> {
    fn process(
        self,
        processor: &CpuImageProcessor,
        options: &ProcessingOptions,
    ) -> Result<Pending<Cpu>, CoreError> {
        process_deferred_cpu(self, processor, options)
    }
}

#[cfg(feature = "vulkan")]
/// Chain preprocess from [`Deferred<Vulkan>`].
pub trait DeferredVulkanProcessExt {
    fn process(
        self,
        processor: &GpuImageProcessor,
        options: &ProcessingOptions,
    ) -> Result<Pending<Vulkan>, CoreError>;
}

#[cfg(feature = "vulkan")]
impl DeferredVulkanProcessExt for Deferred<Vulkan> {
    fn process(
        self,
        processor: &GpuImageProcessor,
        options: &ProcessingOptions,
    ) -> Result<Pending<Vulkan>, CoreError> {
        process_deferred_vulkan(self, processor, options)
    }
}

pub fn process_deferred_cpu(
    deferred: Deferred<Cpu>,
    processor: &CpuImageProcessor,
    options: &ProcessingOptions,
) -> Result<Pending<Cpu>, CoreError> {
    let (device, hardware) = deferred.into_hardware()?;
    let (shape, dtype) = output_shape_dtype(options)?;
    let options = *options;
    let processor = processor.clone();
    Ok(Pending::schedule(
        device,
        shape,
        dtype,
        Box::new(move |target| processor.materialize_from_hardware(hardware, &options, target)),
    ))
}

#[cfg(feature = "vulkan")]
pub fn process_deferred_vulkan(
    deferred: Deferred<Vulkan>,
    processor: &GpuImageProcessor,
    options: &ProcessingOptions,
) -> Result<Pending<Vulkan>, CoreError> {
    if deferred.is_hardware() {
        let (device, hardware) = deferred.into_hardware()?;
        let (shape, dtype) = output_shape_dtype(options)?;
        let options = *options;
        let processor = processor.clone();
        Ok(Pending::schedule(
            device.clone(),
            shape,
            dtype,
            Box::new(move |target| {
                processor.materialize_from_hardware(&device, hardware, &options, target, None)
            }),
        ))
    } else {
        // Keep AHB / custom import deferred until Pending::materialize so camera
        // ImageReader buffers are not held across preprocess scheduling.
        let device = deferred.device().clone();
        let (shape, dtype) = output_shape_dtype(options)?;
        let options = *options;
        let processor = processor.clone();
        Ok(Pending::schedule(
            device,
            shape,
            dtype,
            Box::new(move |target| {
                let image = deferred.materialize()?;
                processor.materialize_image(&image, &options, target, None)
            }),
        ))
    }
}

impl ImageProcessor<Cpu> for CpuImageProcessor {
    fn process(
        &self,
        input: &CpuImage,
        options: &ProcessingOptions,
    ) -> Result<Pending<Cpu>, CoreError> {
        let (shape, dtype) = output_shape_dtype(options)?;
        let hardware = input.hardware();
        let options = *options;
        let processor = self.clone();
        Ok(Pending::schedule(
            Cpu,
            shape,
            dtype,
            Box::new(move |target| processor.materialize_from_hardware(hardware, &options, target)),
        ))
    }
}

pub(crate) fn output_shape_dtype(
    options: &ProcessingOptions,
) -> Result<(infers_core::TensorShape, infers_core::DataType), CoreError> {
    crate::cpu::output_shape_dtype(options)
}

use crate::cpu::CpuImageProcessor;

#[cfg(feature = "vulkan")]
use crate::gpu::GpuImageProcessor;

#[cfg(feature = "vulkan")]
impl ImageProcessor<Vulkan> for GpuImageProcessor {
    fn process(
        &self,
        input: &VulkanImage,
        options: &ProcessingOptions,
    ) -> Result<Pending<Vulkan>, CoreError> {
        GpuImageProcessor::process(self, input, options)
    }
}
