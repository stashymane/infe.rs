use super::build::build_inner;
use super::resources::{drop_gpu_processor_inner, GpuImageProcessorInner, GpuProcessInput};
use super::staging::SessionStagingQuery;
use super::util::{output_shape_dtype, validate_process};
use infers_core::{
    CoreError, HardwareImage, MaterializeTarget, Pending, ProcessingOptions, Tensor,
};
use infers_gpu::{Vulkan, VulkanContext, VulkanImage};
use std::sync::Arc;

/// GPU image processor that dispatches SPIR-V `convert_main` / `convert_image` on a shared context.
#[derive(Clone)]
pub struct GpuImageProcessor {
    inner: Arc<GpuImageProcessorInner>,
}

impl GpuImageProcessor {
    pub fn new(vulkan: Vulkan) -> Result<Self, CoreError> {
        build_inner(vulkan)
            .map(|inner| Self {
                inner: Arc::new(inner),
            })
            .map_err(CoreError::from)
    }

    pub fn vulkan(&self) -> &Vulkan {
        &self.inner.vulkan
    }

    pub fn context(&self) -> &Arc<VulkanContext> {
        &self.inner.context
    }

    pub fn process(
        &self,
        input: &VulkanImage,
        options: &ProcessingOptions,
    ) -> Result<Pending<Vulkan>, CoreError> {
        self.process_with_staging(input, options, None)
    }

    pub(crate) fn process_with_staging(
        &self,
        input: &VulkanImage,
        options: &ProcessingOptions,
        session_staging: Option<SessionStagingQuery>,
    ) -> Result<Pending<Vulkan>, CoreError> {
        validate_process(&self.inner, input, options)?;
        let (shape, dtype) = output_shape_dtype(options)?;
        let process_input = GpuProcessInput::from_image(input)?;
        let mut kernel_opts = *options;
        kernel_opts.src_w = input.width();
        kernel_opts.src_h = input.height();
        kernel_opts.src_format = input.format();

        let inner = Arc::clone(&self.inner);
        let vulkan = inner.vulkan.clone();
        Ok(Pending::schedule(
            vulkan,
            shape,
            dtype,
            Box::new(move |target| {
                inner.materialize(
                    &kernel_opts,
                    process_input,
                    target,
                    session_staging.as_ref(),
                )
            }),
        ))
    }

    pub(crate) fn materialize_from_hardware(
        &self,
        device: &Vulkan,
        hardware: Arc<HardwareImage>,
        options: &ProcessingOptions,
        target: MaterializeTarget,
        session_staging: Option<&SessionStagingQuery>,
    ) -> Result<Tensor<Vulkan>, CoreError> {
        let width = hardware.width();
        let height = hardware.height();
        let format = hardware.format();
        let mut buffer = self
            .inner
            .take_upload_buffer(device, width, height, format)?;
        let result = (|| {
            buffer.write_staging(hardware.as_bytes())?;
            let process_input = GpuProcessInput::linear_unstaged(&buffer)?;

            let mut kernel_opts = *options;
            kernel_opts.src_w = width;
            kernel_opts.src_h = height;
            kernel_opts.src_format = format;

            self.inner.materialize(
                &kernel_opts,
                process_input,
                target,
                session_staging,
            )
        })();
        self.inner.store_upload_buffer(width, height, format, buffer);
        result
    }

    /// Materialize preprocess for an already-placed device image (import happens
    /// inside the caller's Pending closure for deferred AHB paths).
    pub(crate) fn materialize_image(
        &self,
        input: &VulkanImage,
        options: &ProcessingOptions,
        target: MaterializeTarget,
        session_staging: Option<&SessionStagingQuery>,
    ) -> Result<Tensor<Vulkan>, CoreError> {
        validate_process(&self.inner, input, options)?;
        let process_input = GpuProcessInput::from_image(input)?;
        let mut kernel_opts = *options;
        kernel_opts.src_w = input.width();
        kernel_opts.src_h = input.height();
        kernel_opts.src_format = input.format();
        self.inner
            .materialize(&kernel_opts, process_input, target, session_staging)
    }
}

impl Drop for GpuImageProcessor {
    fn drop(&mut self) {
        if Arc::strong_count(&self.inner) != 1 {
            return;
        }
        let inner = Arc::get_mut(&mut self.inner).expect("unique Arc in Drop");
        drop_gpu_processor_inner(inner);
    }
}
