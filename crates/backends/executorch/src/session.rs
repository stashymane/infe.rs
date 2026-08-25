use crate::config::ExecuTorchBackendConfig;
use crate::delegate::ExecuTorchDelegate;
use crate::error::ExecuTorchError;
use crate::program::{validate_program_bytes, MethodDescriptor, TensorDescriptor};
#[cfg(feature = "vulkan")]
use crate::gpu_input::{self, VulkanComputeGraph};
use crate::tensor_ptr::{evalue_to_tensor_buffer, HostInputPlan};
use infers_core::{CoreError, Device, ModelSession, TensorBuffer, TensorShape};
#[cfg(feature = "vulkan")]
use crate::vulkan_adapter;
use executorch::module::Module;
use parking_lot::Mutex;
use std::io::Write;
use std::path::PathBuf;
#[cfg(feature = "vulkan")]
use std::sync::Arc;
use tempfile::NamedTempFile;

#[cfg(feature = "vulkan")]
use infers_gpu::VulkanContext;

pub struct ExecuTorchSession {
    device: Device,
    delegate: ExecuTorchDelegate,
    method: MethodDescriptor,
    input_shapes: Vec<TensorShape>,
    output_shapes: Vec<TensorShape>,
    method_name: String,
    #[cfg(feature = "vulkan")]
    vulkan_context: Option<Arc<VulkanContext>>,
    #[cfg(feature = "vulkan")]
    adapter_registration: Option<vulkan_adapter::ExternalAdapterRegistration>,
    _model_file: NamedTempFile,
    #[cfg(feature = "vulkan")]
    vulkan_graph: Option<VulkanComputeGraph>,
    /// Wrapped in `Option` so [`Drop`] can destroy the Module before clearing
    /// the sticky external adapter (ComputeGraph frees via Adapter VMA).
    module: Option<Mutex<Module<'static>>>,
}

impl Drop for ExecuTorchSession {
    fn drop(&mut self) {
        // Destroy Method/ComputeGraph first while any Vulkan adapter + device remain.
        drop(self.module.take());
        #[cfg(feature = "vulkan")]
        {
            self.vulkan_graph = None;
            // Clear sticky adapter (VMA/caches) while the guard still holds the context.
            drop(self.adapter_registration.take());
            self.vulkan_context = None;
        }
    }
}

impl ExecuTorchSession {
    pub fn load(
        model_bytes: &[u8],
        config: &ExecuTorchBackendConfig,
    ) -> Result<Self, ExecuTorchError> {
        validate_program_bytes(model_bytes)?;
        let method_name = config.method_name().to_string();

        let (device, delegate) = match config {
            ExecuTorchBackendConfig::Xnnpack { .. } => (Device::cpu(), ExecuTorchDelegate::Xnnpack),
            #[cfg(feature = "vulkan")]
            ExecuTorchBackendConfig::Vulkan { context, .. } => (
                context.logical_device().clone(),
                ExecuTorchDelegate::Vulkan,
            ),
        };

        #[cfg(feature = "vulkan")]
        let adapter_registration = match config {
            ExecuTorchBackendConfig::Vulkan { context, .. } => {
                Some(vulkan_adapter::register_external_adapter(context)?)
            }
            ExecuTorchBackendConfig::Xnnpack { .. } => None,
        };

        #[cfg(feature = "vulkan")]
        let vulkan_context = match config {
            ExecuTorchBackendConfig::Vulkan { context, .. } => Some(Arc::clone(context)),
            ExecuTorchBackendConfig::Xnnpack { .. } => None,
        };

        if !delegate.supports_device_kind(device.kind) {
            return Err(ExecuTorchError::Execution(format!(
                "Delegate '{}' is not compatible with device '{:?}'",
                delegate.name(),
                device.kind
            )));
        }

        let mut model_file = NamedTempFile::new()?;
        model_file.write_all(model_bytes)?;
        model_file.flush()?;
        let path: PathBuf = model_file.path().to_path_buf();

        let mut module = Module::new(&path);
        match config {
            ExecuTorchBackendConfig::Xnnpack { num_threads, .. } => {
                use executorch::backend_options::{BackendOption, LoadBackendOptionsMap};
                let threads = BackendOption::new_int("num_threads", *num_threads as i64)
                    .map_err(ExecuTorchError::Native)?;
                let opts = [threads];
                let mut map = LoadBackendOptionsMap::new();
                let _ = map.set_options("XnnpackBackend", &opts);
                module.load(None, Some(&map))?;
            }
            #[cfg(feature = "vulkan")]
            ExecuTorchBackendConfig::Vulkan { .. } => {
                module.load(None, None)?;
            }
        }

        let method_meta = module.method_meta(&method_name).map_err(|_| {
            ExecuTorchError::MethodNotFound(method_name.clone())
        })?;
        let method = MethodDescriptor::from_method_meta(&method_meta)?;
        module.load_method(&method_name, None, None)?;

        #[cfg(feature = "vulkan")]
        let vulkan_graph = VulkanComputeGraph::take_registered();

        let input_shapes = method.inputs.iter().map(|i| i.shape.clone()).collect();
        let output_shapes = method.outputs.iter().map(|o| o.shape.clone()).collect();

        Ok(Self {
            device,
            delegate,
            method,
            input_shapes,
            output_shapes,
            method_name,
            #[cfg(feature = "vulkan")]
            vulkan_context,
            #[cfg(feature = "vulkan")]
            adapter_registration,
            _model_file: model_file,
            #[cfg(feature = "vulkan")]
            vulkan_graph,
            module: Some(Mutex::new(module)),
        })
    }

    pub fn delegate(&self) -> &ExecuTorchDelegate {
        &self.delegate
    }

    pub fn method_name(&self) -> &str {
        &self.method_name
    }

    pub fn input_descriptors(&self) -> &[TensorDescriptor] {
        &self.method.inputs
    }

    pub fn output_descriptors(&self) -> &[TensorDescriptor] {
        &self.method.outputs
    }

    #[cfg(feature = "vulkan")]
    pub fn vulkan_context(&self) -> Option<&Arc<VulkanContext>> {
        self.vulkan_context.as_ref()
    }
}

impl ModelSession for ExecuTorchSession {
    fn device(&self) -> &Device {
        &self.device
    }

    fn input_shapes(&self) -> &[TensorShape] {
        &self.input_shapes
    }

    fn output_shapes(&self) -> &[TensorShape] {
        &self.output_shapes
    }

    fn run(
        &mut self,
        inputs: &[&dyn TensorBuffer],
    ) -> Result<Vec<Box<dyn TensorBuffer>>, CoreError> {
        if inputs.len() != self.method.inputs.len() {
            return Err(CoreError::InferenceFailed(format!(
                "Input count mismatch for method '{}': expected {}, received {}",
                self.method.name,
                self.method.inputs.len(),
                inputs.len()
            )));
        }

        for (i, (&input, expected)) in inputs.iter().zip(self.method.inputs.iter()).enumerate() {
            if input.device().kind != self.device.kind {
                return Err(CoreError::DeviceMismatch {
                    expected: self.device.clone(),
                    actual: input.device().clone(),
                });
            }
            if input.shape() != &expected.shape {
                return Err(CoreError::InvalidShape(format!(
                    "Input {} ('{}') shape mismatch: expected {:?}, got {:?}",
                    i,
                    expected.name,
                    expected.shape.dims(),
                    input.shape().dims()
                )));
            }
            if input.dtype() != expected.dtype {
                return Err(CoreError::InvalidDataType {
                    expected: expected.dtype,
                    actual: input.dtype(),
                });
            }
        }

        let mut module = self
            .module
            .as_ref()
            .ok_or_else(|| {
                CoreError::InferenceFailed("ExecuTorch session module already dropped".into())
            })?
            .lock();

        #[cfg(feature = "vulkan")]
        let outputs = if self.delegate == ExecuTorchDelegate::Vulkan {
            let context = self.vulkan_context.as_ref().ok_or_else(|| {
                CoreError::InferenceFailed("Vulkan session missing shared VulkanContext".into())
            })?;
            let plan =
                gpu_input::prepare_inputs(inputs, context, self.vulkan_graph)?;
            gpu_input::set_skip_staging_copy_mask(plan.skip_staging_mask);
            let evalues: Vec<executorch::evalue::EValue<'_>> = plan
                .tensor_ptrs
                .iter()
                .map(|p| p.as_evalue())
                .collect();
            let result = module
                .execute(&self.method_name, &evalues)
                .map_err(|e| {
                    CoreError::InferenceFailed(format!("ExecuTorch execute failed: {:?}", e))
                });
            gpu_input::set_skip_staging_copy_mask(0);
            result?
        } else {
            run_host_path(&mut module, &self.method_name, inputs)?
        };

        #[cfg(not(feature = "vulkan"))]
        let outputs = run_host_path(&mut module, &self.method_name, inputs)?;

        let mut owned: Vec<Box<dyn TensorBuffer>> = Vec::with_capacity(self.method.outputs.len());
        for (out_evalue, out_desc) in outputs.iter().zip(self.method.outputs.iter()) {
            let tb = evalue_to_tensor_buffer(out_evalue, &self.device, out_desc)
                .map_err(CoreError::from)?;
            owned.push(Box::new(tb));
        }
        Ok(owned)
    }
}

fn run_host_path<'a>(
    module: &'a mut Module<'static>,
    method_name: &str,
    inputs: &[&dyn TensorBuffer],
) -> Result<Vec<executorch::evalue::EValue<'a>>, CoreError> {
    let plan = HostInputPlan::build(inputs)?;
    let evalues: Vec<_> = plan.tensor_ptrs.iter().map(|p| p.as_evalue()).collect();
    module
        .execute(method_name, &evalues)
        .map_err(|e| CoreError::InferenceFailed(format!("ExecuTorch execute failed: {:?}", e)))
}
