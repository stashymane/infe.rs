use crate::config::XnnpackOptions;
#[cfg(feature = "vulkan")]
use crate::config::VulkanOptions;
use crate::delegate::ExecuTorchDelegate;
use crate::error::ExecuTorchError;
use crate::program::{validate_program_bytes, MethodDescriptor, TensorDescriptor};
#[cfg(feature = "vulkan")]
use crate::gpu_input::{self, VulkanComputeGraph};
use crate::tensor_ptr::{evalue_to_cpu_tensor, HostInputPlan};
use infers_core::{CoreError, Cpu, Device, MaterializeTarget, Pending, Session, SessionInputSink, Tensor, TensorShape, prepare_infer};
use infers_core::infer_input::InferInput;
#[cfg(feature = "vulkan")]
use crate::vulkan_adapter;
use executorch::module::Module;
use parking_lot::Mutex;
use std::io::Write;
use std::path::Path;
#[cfg(feature = "vulkan")]
use std::sync::Arc;
use tempfile::NamedTempFile;

#[cfg(feature = "vulkan")]
use infers_gpu::{Vulkan, VulkanContext};

#[cfg(feature = "vulkan")]
struct VulkanExtra {
    context: Arc<VulkanContext>,
    adapter_registration: vulkan_adapter::ExternalAdapterRegistration,
    vulkan_graph: Option<VulkanComputeGraph>,
}

pub struct ExecuTorchSession<D: Device> {
    device: D,
    delegate: ExecuTorchDelegate,
    method: MethodDescriptor,
    input_shapes: Vec<TensorShape>,
    output_shapes: Vec<TensorShape>,
    method_name: String,
    _model_file: Option<NamedTempFile>,
    #[cfg(feature = "vulkan")]
    vulkan_extra: Option<VulkanExtra>,
    module: Option<Mutex<Module<'static>>>,
}

impl<D: Device> Drop for ExecuTorchSession<D> {
    fn drop(&mut self) {
        drop(self.module.take());
        #[cfg(feature = "vulkan")]
        if let Some(extra) = self.vulkan_extra.take() {
            drop(extra.adapter_registration);
        }
    }
}

impl ExecuTorchSession<Cpu> {
    pub fn load_xnnpack(
        model_bytes: &[u8],
        opts: &XnnpackOptions,
    ) -> Result<Self, ExecuTorchError> {
        validate_program_bytes(model_bytes)?;
        let mut model_file = NamedTempFile::new()?;
        model_file.write_all(model_bytes)?;
        model_file.flush()?;
        let path = model_file.path().to_path_buf();
        Self::load_xnnpack_from_path_inner(&path, opts, Some(model_file))
    }

    pub fn load_xnnpack_from_path(
        path: &Path,
        opts: &XnnpackOptions,
    ) -> Result<Self, ExecuTorchError> {
        use std::io::Read;
        let mut file = std::fs::File::open(path).map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!("Failed to open model file '{}': {}", path.display(), e),
            )
        })?;
        let mut header = [0u8; 64];
        let n = file.read(&mut header)?;
        validate_program_bytes(&header[..n])?;
        Self::load_xnnpack_from_path_inner(path, opts, None)
    }

    fn load_xnnpack_from_path_inner(
        path: &Path,
        opts: &XnnpackOptions,
        model_file: Option<NamedTempFile>,
    ) -> Result<Self, ExecuTorchError> {
        let method_name = opts.method_name().to_string();
        let mut module = Module::new(path);
        use executorch::backend_options::{BackendOption, LoadBackendOptionsMap};
        let threads = BackendOption::new_int("num_threads", opts.num_threads as i64)
            .map_err(ExecuTorchError::Native)?;
        let thread_opts = [threads];
        let mut map = LoadBackendOptionsMap::new();
        let _ = map.set_options("XnnpackBackend", &thread_opts);
        module.load(None, Some(&map))?;

        let method_meta = module
            .method_meta(&method_name)
            .map_err(|_| ExecuTorchError::MethodNotFound(method_name.clone()))?;
        let method = MethodDescriptor::from_method_meta(&method_meta)?;
        module.load_method(&method_name, None, None)?;

        let input_shapes = method.inputs.iter().map(|i| i.shape.clone()).collect();
        let output_shapes = method.outputs.iter().map(|o| o.shape.clone()).collect();

        Ok(Self {
            device: Cpu,
            delegate: ExecuTorchDelegate::Xnnpack,
            method,
            input_shapes,
            output_shapes,
            method_name,
            _model_file: model_file,
            #[cfg(feature = "vulkan")]
            vulkan_extra: None,
            module: Some(Mutex::new(module)),
        })
    }
}

#[cfg(feature = "vulkan")]
impl ExecuTorchSession<Vulkan> {
    pub fn load_vulkan(
        model_bytes: &[u8],
        device: &Vulkan,
        opts: &VulkanOptions,
    ) -> Result<Self, ExecuTorchError> {
        validate_program_bytes(model_bytes)?;
        let mut model_file = NamedTempFile::new()?;
        model_file.write_all(model_bytes)?;
        model_file.flush()?;
        let path = model_file.path().to_path_buf();
        Self::load_vulkan_from_path_inner(&path, device, opts, Some(model_file))
    }

    pub fn load_vulkan_from_path(
        path: &Path,
        device: &Vulkan,
        opts: &VulkanOptions,
    ) -> Result<Self, ExecuTorchError> {
        use std::io::Read;
        let mut file = std::fs::File::open(path).map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!("Failed to open model file '{}': {}", path.display(), e),
            )
        })?;
        let mut header = [0u8; 64];
        let n = file.read(&mut header)?;
        validate_program_bytes(&header[..n])?;
        Self::load_vulkan_from_path_inner(path, device, opts, None)
    }

    fn load_vulkan_from_path_inner(
        path: &Path,
        device: &Vulkan,
        opts: &VulkanOptions,
        model_file: Option<NamedTempFile>,
    ) -> Result<Self, ExecuTorchError> {
        let method_name = opts.method_name().to_string();
        let context = Arc::clone(device.context());
        let adapter_registration = vulkan_adapter::register_external_adapter(&context)?;
        let mut module = Module::new(path);
        module.load(None, None)?;

        let method_meta = module
            .method_meta(&method_name)
            .map_err(|_| ExecuTorchError::MethodNotFound(method_name.clone()))?;
        let method = MethodDescriptor::from_method_meta(&method_meta)?;
        module.load_method(&method_name, None, None)?;

        let vulkan_graph = VulkanComputeGraph::take_registered();
        let input_shapes = method.inputs.iter().map(|i| i.shape.clone()).collect();
        let output_shapes = method.outputs.iter().map(|o| o.shape.clone()).collect();

        Ok(Self {
            device: device.clone(),
            delegate: ExecuTorchDelegate::Vulkan,
            method,
            input_shapes,
            output_shapes,
            method_name,
            _model_file: model_file,
            vulkan_extra: Some(VulkanExtra {
                context,
                adapter_registration,
                vulkan_graph,
            }),
            module: Some(Mutex::new(module)),
        })
    }

    pub fn vulkan(&self) -> &Vulkan {
        &self.device
    }
}

impl<D: Device> ExecuTorchSession<D> {
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
}

impl Session<Cpu> for ExecuTorchSession<Cpu> {
    type Output = Tensor<Cpu>;

    fn device(&self) -> &Cpu {
        &self.device
    }

    fn input_shapes(&self) -> &[TensorShape] {
        &self.input_shapes
    }

    fn output_shapes(&self) -> &[TensorShape] {
        &self.output_shapes
    }

    fn infer(&mut self, input: impl InferInput<Cpu>) -> Result<Vec<Tensor<Cpu>>, CoreError> {
        let mut sink = SingleInputSink::<Cpu>::default();
        prepare_infer(input, &mut sink)?;
        let tensor = sink
            .tensor
            .ok_or_else(|| CoreError::InferenceFailed("missing inference input".into()))?;
        validate_inputs(&self.method, &[&tensor])?;

        let mut module = self
            .module
            .as_ref()
            .ok_or_else(|| CoreError::InferenceFailed("session module dropped".into()))?
            .lock();

        let plan = HostInputPlan::build(&[&tensor])?;
        let evalues: Vec<_> = plan.tensor_ptrs.iter().map(|p| p.as_evalue()).collect();
        let outputs = module
            .execute(&self.method_name, &evalues)
            .map_err(|e| CoreError::InferenceFailed(format!("ExecuTorch execute failed: {e:?}")))?;

        let mut owned = Vec::with_capacity(self.method.outputs.len());
        for (out_evalue, out_desc) in outputs.iter().zip(self.method.outputs.iter()) {
            owned.push(evalue_to_cpu_tensor(out_evalue, out_desc).map_err(CoreError::from)?);
        }
        Ok(owned)
    }
}

#[cfg(feature = "vulkan")]
impl Session<Vulkan> for ExecuTorchSession<Vulkan> {
    type Output = Tensor<Cpu>;

    fn device(&self) -> &Vulkan {
        &self.device
    }

    fn input_shapes(&self) -> &[TensorShape] {
        &self.input_shapes
    }

    fn output_shapes(&self) -> &[TensorShape] {
        &self.output_shapes
    }

    fn infer(&mut self, input: impl InferInput<Vulkan>) -> Result<Vec<Tensor<Cpu>>, CoreError> {
        let extra = self.vulkan_extra.as_ref().ok_or_else(|| {
            CoreError::InferenceFailed("Vulkan session missing context".into())
        })?;

        let mut sink = VulkanInferSink {
            graph: extra.vulkan_graph,
            tensor: None,
        };
        prepare_infer(input, &mut sink)?;
        let tensor = sink
            .tensor
            .ok_or_else(|| CoreError::InferenceFailed("missing inference input".into()))?;
        validate_inputs_vulkan(&self.method, &[&tensor], &self.device)?;

        let mut module = self
            .module
            .as_ref()
            .ok_or_else(|| CoreError::InferenceFailed("session module dropped".into()))?
            .lock();

        let plan = gpu_input::prepare_inputs(&[&tensor], &extra.context, extra.vulkan_graph)?;
        gpu_input::set_skip_staging_copy_mask(plan.skip_staging_mask);
        let evalues = plan.evalues();
        let outputs = module
            .execute(&self.method_name, &evalues)
            .map_err(|e| CoreError::InferenceFailed(format!("ExecuTorch execute failed: {e:?}")));
        gpu_input::set_skip_staging_copy_mask(0);
        let outputs = outputs?;

        let mut owned = Vec::with_capacity(self.method.outputs.len());
        for (out_evalue, out_desc) in outputs.iter().zip(self.method.outputs.iter()) {
            owned.push(evalue_to_cpu_tensor(out_evalue, out_desc).map_err(CoreError::from)?);
        }
        Ok(owned)
    }
}

fn validate_inputs(
    method: &MethodDescriptor,
    inputs: &[&Tensor<Cpu>],
) -> Result<(), CoreError> {
    if inputs.len() != method.inputs.len() {
        return Err(CoreError::InferenceFailed(format!(
            "Input count mismatch: expected {}, received {}",
            method.inputs.len(),
            inputs.len()
        )));
    }
    for (i, (&input, expected)) in inputs.iter().zip(method.inputs.iter()).enumerate() {
        if input.shape() != &expected.shape {
            return Err(CoreError::InvalidShape(format!(
                "Input {} shape mismatch: expected {:?}, got {:?}",
                i,
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
    Ok(())
}

#[derive(Default)]
struct SingleInputSink<D: Device> {
    tensor: Option<Tensor<D>>,
}

impl<D: Device> SessionInputSink<D> for SingleInputSink<D> {
    fn materialize_pending(&mut self, pending: Pending<D>) -> Result<(), CoreError> {
        self.tensor = Some(pending.materialize()?);
        Ok(())
    }

    fn adopt_tensor(&mut self, tensor: &Tensor<D>) -> Result<(), CoreError> {
        self.tensor = Some(tensor.clone());
        Ok(())
    }
}

#[cfg(feature = "vulkan")]
struct VulkanInferSink {
    graph: Option<gpu_input::VulkanComputeGraph>,
    tensor: Option<Tensor<Vulkan>>,
}

#[cfg(feature = "vulkan")]
impl SessionInputSink<Vulkan> for VulkanInferSink {
    fn materialize_pending(&mut self, pending: Pending<Vulkan>) -> Result<(), CoreError> {
        use infers_processing::gpu::{set_session_staging_query, SessionStagingQuery};
        use std::sync::Arc;

        let graph = self.graph.ok_or_else(|| {
            CoreError::InferenceFailed("Vulkan compute graph unavailable for session input".into())
        })?;
        let query: SessionStagingQuery = Arc::new(move |slot: usize| {
            graph
                .staging_target(slot)
                .map(|(handle, _size)| handle)
                .map_err(|e| CoreError::BufferTransferFailed(e.to_string()))
        });
        set_session_staging_query(Some(Arc::clone(&query)));
        let result = pending.materialize_with_target(MaterializeTarget::SessionInput { slot: 0 });
        set_session_staging_query(None);
        self.tensor = Some(result?);
        Ok(())
    }

    fn adopt_tensor(&mut self, tensor: &Tensor<Vulkan>) -> Result<(), CoreError> {
        self.tensor = Some(tensor.clone());
        Ok(())
    }
}

#[cfg(feature = "vulkan")]
fn validate_inputs_vulkan(
    method: &MethodDescriptor,
    inputs: &[&Tensor<Vulkan>],
    device: &Vulkan,
) -> Result<(), CoreError> {
    if inputs.len() != method.inputs.len() {
        return Err(CoreError::InferenceFailed(format!(
            "Input count mismatch: expected {}, received {}",
            method.inputs.len(),
            inputs.len()
        )));
    }
    for (i, input) in inputs.iter().enumerate() {
        if !Arc::ptr_eq(device.context(), input.device().context()) {
            return Err(CoreError::DeviceMismatch {
                expected: device.info().clone(),
                actual: input.device().info().clone(),
            });
        }
        let expected = &method.inputs[i];
        if input.shape() != &expected.shape {
            return Err(CoreError::InvalidShape(format!(
                "Input {} shape mismatch",
                i
            )));
        }
        if input.dtype() != expected.dtype {
            return Err(CoreError::InvalidDataType {
                expected: expected.dtype,
                actual: input.dtype(),
            });
        }
    }
    Ok(())
}
