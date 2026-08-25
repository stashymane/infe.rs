use crate::config::ExecuTorchBackendConfig;
use crate::delegate::ExecuTorchDelegate;
use crate::error::ExecuTorchError;
use crate::program::{MethodDescriptor, ProgramMetadata, TensorDescriptor};
#[cfg(feature = "vulkan")]
use crate::gpu_input::{self, VulkanComputeGraph};
use infers_core::{CoreError, Device, ModelSession, TensorBuffer, TensorShape};
#[cfg(feature = "vulkan")]
use crate::vulkan_adapter;
use executorch::evalue::IntoEValue;
use executorch::module::Module;
use executorch::ndarray;
use executorch::tensor::TensorPtr;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
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
    _model_file: NamedTempFile,
    module: Mutex<Module<'static>>,
    method_name: String,
    #[cfg(feature = "vulkan")]
    vulkan_context: Option<Arc<VulkanContext>>,
    #[cfg(feature = "vulkan")]
    vulkan_graph: Option<VulkanComputeGraph>,
}

impl ExecuTorchSession {
    pub fn load(
        model_bytes: &[u8],
        config: &ExecuTorchBackendConfig,
    ) -> Result<Self, ExecuTorchError> {
        let metadata = ProgramMetadata::from_bytes(model_bytes)?;
        let method_name = config.method_name().to_string();
        let method = metadata
            .method(&method_name)
            .cloned()
            .ok_or_else(|| ExecuTorchError::MethodNotFound(method_name.clone()))?;

        let (device, delegate) = match config {
            ExecuTorchBackendConfig::Xnnpack { .. } => (Device::cpu(), ExecuTorchDelegate::Xnnpack),
            #[cfg(feature = "vulkan")]
            ExecuTorchBackendConfig::Vulkan { context, .. } => {
                vulkan_adapter::register_external_adapter(context)?;
                (
                    context.logical_device().clone(),
                    ExecuTorchDelegate::Vulkan,
                )
            }
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
            _model_file: model_file,
            module: Mutex::new(module),
            method_name,
            #[cfg(feature = "vulkan")]
            vulkan_context,
            #[cfg(feature = "vulkan")]
            vulkan_graph,
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

        let mut module = self.module.lock().map_err(|_| {
            CoreError::InferenceFailed("ExecuTorch module lock poisoned".into())
        })?;

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
                .map(|p| p.into_evalue())
                .collect();
            let result = module
                .execute(&self.method_name, &evalues)
                .map_err(|e| {
                    CoreError::InferenceFailed(format!("ExecuTorch execute failed: {:?}", e))
                });
            gpu_input::set_skip_staging_copy_mask(0);
            result?
        } else {
            let host_inputs = collect_host_inputs(inputs)?;
            let owned_ptrs = host_inputs
                .iter()
                .map(host_to_tensor_ptr)
                .collect::<Result<Vec<_>, _>>()?;
            let evalues: Vec<executorch::evalue::EValue<'_>> = owned_ptrs
                .iter()
                .map(|ptr| match ptr {
                    OwnedTensorPtr::F32(p) => p.into_evalue(),
                    OwnedTensorPtr::U8(p) => p.into_evalue(),
                    OwnedTensorPtr::I32(p) => p.into_evalue(),
                    OwnedTensorPtr::I64(p) => p.into_evalue(),
                })
                .collect();
            module
                .execute(&self.method_name, &evalues)
                .map_err(|e| CoreError::InferenceFailed(format!("ExecuTorch execute failed: {:?}", e)))?
        };

        #[cfg(not(feature = "vulkan"))]
        let outputs = {
            let host_inputs = collect_host_inputs(inputs)?;
            let owned_ptrs = host_inputs
                .iter()
                .map(host_to_tensor_ptr)
                .collect::<Result<Vec<_>, _>>()?;
            let evalues: Vec<executorch::evalue::EValue<'_>> = owned_ptrs
                .iter()
                .map(|ptr| match ptr {
                    OwnedTensorPtr::F32(p) => p.into_evalue(),
                    OwnedTensorPtr::U8(p) => p.into_evalue(),
                    OwnedTensorPtr::I32(p) => p.into_evalue(),
                    OwnedTensorPtr::I64(p) => p.into_evalue(),
                })
                .collect();
            module
                .execute(&self.method_name, &evalues)
                .map_err(|e| CoreError::InferenceFailed(format!("ExecuTorch execute failed: {:?}", e)))?
        };

        let mut owned: Vec<Box<dyn TensorBuffer>> = Vec::with_capacity(self.method.outputs.len());
        for (out_evalue, out_desc) in outputs.iter().zip(self.method.outputs.iter()) {
            #[cfg(feature = "vulkan")]
            let tb = gpu_input::evalue_to_tensor_buffer(out_evalue, &self.device, out_desc)
                .map_err(CoreError::from)?;
            #[cfg(not(feature = "vulkan"))]
            let tb = evalue_to_tensor_buffer(out_evalue, &self.device, out_desc)
                .map_err(CoreError::from)?;
            owned.push(Box::new(tb));
        }
        Ok(owned)
    }
}

struct HostInput {
    dtype: infers_core::DataType,
    shape: TensorShape,
    bytes: Vec<u8>,
}

enum OwnedTensorPtr {
    F32(TensorPtr<'static, executorch::tensor::View<f32>>),
    U8(TensorPtr<'static, executorch::tensor::View<u8>>),
    I32(TensorPtr<'static, executorch::tensor::View<i32>>),
    I64(TensorPtr<'static, executorch::tensor::View<i64>>),
}

fn collect_host_inputs(inputs: &[&dyn TensorBuffer]) -> Result<Vec<HostInput>, CoreError> {
    let mut out = Vec::with_capacity(inputs.len());
    for input in inputs {
        let host = input.read_to_cpu()?;
        out.push(HostInput {
            dtype: host.dtype(),
            shape: host.shape().clone(),
            bytes: host.as_bytes().to_vec(),
        });
    }
    Ok(out)
}

fn host_to_tensor_ptr(host: &HostInput) -> Result<OwnedTensorPtr, CoreError> {
    let dims: Vec<usize> = host.shape.dims().to_vec();
    match host.dtype {
        infers_core::DataType::F32 => {
            let data = bytes_as_vec_f32(&host.bytes, host.shape.element_count())?;
            let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&dims), data)
                .map_err(|e| CoreError::InferenceFailed(e.to_string()))?;
            Ok(OwnedTensorPtr::F32(
                TensorPtr::from_array(arr).map_err(|e| {
                    CoreError::InferenceFailed(format!("TensorPtr::from_array failed: {:?}", e))
                })?,
            ))
        }
        infers_core::DataType::U8 => {
            let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&dims), host.bytes.clone())
                .map_err(|e| CoreError::InferenceFailed(e.to_string()))?;
            Ok(OwnedTensorPtr::U8(
                TensorPtr::from_array(arr).map_err(|e| {
                    CoreError::InferenceFailed(format!("TensorPtr::from_array failed: {:?}", e))
                })?,
            ))
        }
        infers_core::DataType::I32 => {
            let data = bytes_as_vec_i32(&host.bytes, host.shape.element_count())?;
            let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&dims), data)
                .map_err(|e| CoreError::InferenceFailed(e.to_string()))?;
            Ok(OwnedTensorPtr::I32(
                TensorPtr::from_array(arr).map_err(|e| {
                    CoreError::InferenceFailed(format!("TensorPtr::from_array failed: {:?}", e))
                })?,
            ))
        }
        infers_core::DataType::I64 => {
            let data = bytes_as_vec_i64(&host.bytes, host.shape.element_count())?;
            let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&dims), data)
                .map_err(|e| CoreError::InferenceFailed(e.to_string()))?;
            Ok(OwnedTensorPtr::I64(
                TensorPtr::from_array(arr).map_err(|e| {
                    CoreError::InferenceFailed(format!("TensorPtr::from_array failed: {:?}", e))
                })?,
            ))
        }
        other => Err(CoreError::InferenceFailed(format!(
            "Unsupported input dtype for ExecuTorch Module: {:?}",
            other
        ))),
    }
}

fn bytes_as_vec_f32(bytes: &[u8], count: usize) -> Result<Vec<f32>, CoreError> {
    if bytes.len() < count * 4 {
        return Err(CoreError::BufferTransferFailed(
            "f32 buffer too small".into(),
        ));
    }
    let mut out = Vec::with_capacity(count);
    for chunk in bytes.chunks_exact(4).take(count) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn bytes_as_vec_i32(bytes: &[u8], count: usize) -> Result<Vec<i32>, CoreError> {
    if bytes.len() < count * 4 {
        return Err(CoreError::BufferTransferFailed(
            "i32 buffer too small".into(),
        ));
    }
    let mut out = Vec::with_capacity(count);
    for chunk in bytes.chunks_exact(4).take(count) {
        out.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn bytes_as_vec_i64(bytes: &[u8], count: usize) -> Result<Vec<i64>, CoreError> {
    if bytes.len() < count * 8 {
        return Err(CoreError::BufferTransferFailed(
            "i64 buffer too small".into(),
        ));
    }
    let mut out = Vec::with_capacity(count);
    for chunk in bytes.chunks_exact(8).take(count) {
        out.push(i64::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(out)
}

#[cfg(not(feature = "vulkan"))]
fn evalue_to_tensor_buffer(
    value: &executorch::evalue::EValue<'_>,
    device: &Device,
    desc: &TensorDescriptor,
) -> Result<crate::tensor::ExecuTorchTensorBuffer, ExecuTorchError> {
    use crate::tensor::scalar_type_to_data_type;
    use executorch::evalue::Tag;
    if value.tag() != Tag::Tensor {
        return Err(ExecuTorchError::Execution(format!(
            "Expected tensor output for '{}', got {:?}",
            desc.name,
            value.tag()
        )));
    }
    let tensor = value.as_tensor();
    let nbytes = tensor.nbytes();
    let ptr = tensor.as_data_ptr_raw() as *const u8;
    if ptr.is_null() && nbytes > 0 {
        return Err(ExecuTorchError::BufferError(
            "Output tensor has null data pointer".into(),
        ));
    }
    let bytes = if nbytes == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(ptr, nbytes).to_vec() }
    };

    let sizes = tensor.sizes();
    let shape = TensorShape::new(sizes.iter().map(|&d| d as usize).collect::<Vec<_>>())?;
    let dtype = scalar_type_to_data_type(tensor.scalar_type())?;

    crate::tensor::ExecuTorchTensorBuffer::new(device.clone(), shape, dtype, bytes)
}
