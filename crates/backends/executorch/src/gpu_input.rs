//! GPU-direct ExecuTorch Vulkan input path (Phase 1).

use crate::error::ExecuTorchError;
use crate::tensor::{ExecuTorchTensorBuffer, scalar_type_to_data_type};
use executorch::evalue::{EValue, IntoEValue, Tag};
use executorch::ndarray;
use executorch::tensor::{TensorPtr, TensorPtrBuilder, View};
use infers_core::{CoreError, DataType, Device, TensorBuffer, TensorShape};
use infers_gpu::{VulkanBufferHandle, VulkanContext};
use processing_gpu::GpuTensorBuffer;
use std::sync::Arc;

#[link(name = "infers_et_vulkan_ffi")]
unsafe extern "C" {
    fn infers_et_take_registered_vulkan_graph() -> *mut std::ffi::c_void;
    fn infers_et_vulkan_set_skip_staging_copy_mask(mask: u64);
    fn infers_et_vulkan_input_staging_buffer(
        graph_ptr: *mut std::ffi::c_void,
        input_index: u32,
        out_buffer: *mut infers_gpu::ash::vk::Buffer,
        out_offset: *mut u64,
        out_size: *mut u64,
    ) -> i32;
}

/// Handle to a Vulkan-delegated model's `ComputeGraph`, captured after `load_method`.
#[derive(Clone, Copy)]
pub struct VulkanComputeGraph {
    ptr: *mut std::ffi::c_void,
}

unsafe impl Send for VulkanComputeGraph {}
unsafe impl Sync for VulkanComputeGraph {}

impl VulkanComputeGraph {
    pub fn take_registered() -> Option<Self> {
        let ptr = unsafe { infers_et_take_registered_vulkan_graph() };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    fn staging_target(&self, input_index: usize) -> Result<(VulkanBufferHandle, u64), ExecuTorchError> {
        let mut buffer = infers_gpu::ash::vk::Buffer::null();
        let mut offset = 0u64;
        let mut size = 0u64;
        let rc = unsafe {
            infers_et_vulkan_input_staging_buffer(
                self.ptr,
                input_index as u32,
                &mut buffer,
                &mut offset,
                &mut size,
            )
        };
        if rc != 0 {
            return Err(ExecuTorchError::Execution(format!(
                "Failed to query ET-VK staging buffer for input {input_index}: code {rc}"
            )));
        }
        Ok((
            VulkanBufferHandle {
                buffer,
                memory: infers_gpu::ash::vk::DeviceMemory::null(),
                offset,
                size,
            },
            size,
        ))
    }
}

struct HostVisibleInput {
    buffer: infers_gpu::AllocatedBuffer,
    shape: TensorShape,
    dtype: DataType,
}

impl HostVisibleInput {
    fn new(context: &VulkanContext, shape: TensorShape, dtype: DataType) -> Result<Self, ExecuTorchError> {
        use infers_gpu::ash::vk;
        use infers_gpu::gpu_allocator::MemoryLocation;
        let bytes = shape.byte_size(dtype) as u64;
        let buffer = context
            .create_buffer(
                bytes,
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::CpuToGpu,
                "et-host-visible-input",
            )
            .map_err(|err| ExecuTorchError::Execution(err.to_string()))?;
        Ok(Self {
            buffer,
            shape,
            dtype,
        })
    }

    fn copy_from_gpu(
        &self,
        context: &VulkanContext,
        src: VulkanBufferHandle,
    ) -> Result<(), ExecuTorchError> {
        let copy_size = src.size.min(self.buffer.size);
        context
            .copy_buffer(src.buffer, src.offset, self.buffer.buffer, 0, copy_size)
            .map_err(|err| ExecuTorchError::Execution(err.to_string()))
    }

    fn tensor_ptr(&self) -> Result<OwnedTensorPtr, CoreError> {
        let mapped = self
            .buffer
            .allocation
            .mapped_slice()
            .ok_or_else(|| {
                CoreError::BufferTransferFailed("ET host-visible input is not mapped".into())
            })?;
        let dims: Vec<i32> = self
            .shape
            .dims()
            .iter()
            .map(|&d| d as i32)
            .collect();
        match self.dtype {
            DataType::F32 => {
                let ptr = mapped.as_ptr() as *const f32;
                let ptr = unsafe {
                    TensorPtrBuilder::<View<f32>>::from_ptr(ptr, dims.clone())
                        .build()
                        .map_err(|e| {
                            CoreError::InferenceFailed(format!("TensorPtr build failed: {e:?}"))
                        })?
                };
                Ok(OwnedTensorPtr::F32(ptr))
            }
            DataType::U8 => {
                let ptr = mapped.as_ptr();
                let ptr = unsafe {
                    TensorPtrBuilder::<View<u8>>::from_ptr(ptr, dims.clone())
                        .build()
                        .map_err(|e| {
                            CoreError::InferenceFailed(format!("TensorPtr build failed: {e:?}"))
                        })?
                };
                Ok(OwnedTensorPtr::U8(ptr))
            }
            other => Err(CoreError::InferenceFailed(format!(
                "Unsupported GPU input dtype for ExecuTorch: {other:?}"
            ))),
        }
    }
}

pub enum OwnedTensorPtr {
    F32(TensorPtr<'static, View<f32>>),
    U8(TensorPtr<'static, View<u8>>),
    I32(TensorPtr<'static, View<i32>>),
    I64(TensorPtr<'static, View<i64>>),
}

impl OwnedTensorPtr {
    pub fn into_evalue(&self) -> EValue<'_> {
        match self {
            Self::F32(p) => p.into_evalue(),
            Self::U8(p) => p.into_evalue(),
            Self::I32(p) => p.into_evalue(),
            Self::I64(p) => p.into_evalue(),
        }
    }
}

pub struct GpuInputPlan {
    pub tensor_ptrs: Vec<OwnedTensorPtr>,
    pub skip_staging_mask: u64,
    _host_fallback: Vec<HostVisibleInput>,
    _gpu_pins: Vec<GpuTensorBuffer>,
}

pub fn prepare_inputs(
    inputs: &[&dyn TensorBuffer],
    context: &Arc<VulkanContext>,
    vulkan_graph: Option<VulkanComputeGraph>,
) -> Result<GpuInputPlan, CoreError> {
    let mut tensor_ptrs = Vec::with_capacity(inputs.len());
    let mut skip_staging_mask = 0u64;
    let mut host_fallback = Vec::new();
    let mut gpu_pins = Vec::new();

    for (index, input) in inputs.iter().enumerate() {
        if let Some(gpu) = input.as_any().downcast_ref::<GpuTensorBuffer>() {
            let src = gpu.vulkan_handle().ok_or_else(|| {
                CoreError::BufferTransferFailed("GPU tensor buffer already destroyed".into())
            })?;
            gpu_pins.push(gpu.clone());

            if let Some(graph) = vulkan_graph {
                if let Ok((dst, staging_size)) = graph.staging_target(index) {
                    let copy_size = src.size.min(staging_size);
                    context
                        .copy_buffer(
                            src.buffer,
                            src.offset,
                            dst.buffer,
                            dst.offset,
                            copy_size,
                        )
                        .map_err(|err| CoreError::BufferTransferFailed(err.to_string()))?;
                    skip_staging_mask |= 1u64 << index;
                    tensor_ptrs.push(placeholder_tensor_ptr(*input)?);
                    continue;
                }
            }

            let host = HostVisibleInput::new(context, input.shape().clone(), input.dtype())?;
            host.copy_from_gpu(context, src)
                .map_err(|err| CoreError::BufferTransferFailed(err.to_string()))?;
            tensor_ptrs.push(host.tensor_ptr()?);
            host_fallback.push(host);
            continue;
        }

        tensor_ptrs.push(cpu_tensor_ptr(*input)?);
    }

    Ok(GpuInputPlan {
        tensor_ptrs,
        skip_staging_mask,
        _host_fallback: host_fallback,
        _gpu_pins: gpu_pins,
    })
}

pub fn set_skip_staging_copy_mask(mask: u64) {
    unsafe { infers_et_vulkan_set_skip_staging_copy_mask(mask) };
}

fn placeholder_tensor_ptr(input: &dyn TensorBuffer) -> Result<OwnedTensorPtr, CoreError> {
    let count = input.shape().element_count();
    match input.dtype() {
        DataType::F32 => {
            let data = vec![0.0f32; count];
            let dims: Vec<usize> = input.shape().dims().to_vec();
            let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&dims), data)
                .map_err(|e| CoreError::InferenceFailed(e.to_string()))?;
            Ok(OwnedTensorPtr::F32(
                TensorPtr::from_array(arr).map_err(|e| {
                    CoreError::InferenceFailed(format!("TensorPtr::from_array failed: {:?}", e))
                })?,
            ))
        }
        DataType::U8 => {
            let data = vec![0u8; count];
            let dims: Vec<usize> = input.shape().dims().to_vec();
            let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&dims), data)
                .map_err(|e| CoreError::InferenceFailed(e.to_string()))?;
            Ok(OwnedTensorPtr::U8(
                TensorPtr::from_array(arr).map_err(|e| {
                    CoreError::InferenceFailed(format!("TensorPtr::from_array failed: {:?}", e))
                })?,
            ))
        }
        other => Err(CoreError::InferenceFailed(format!(
            "Unsupported placeholder dtype: {other:?}"
        ))),
    }
}

fn cpu_tensor_ptr(input: &dyn TensorBuffer) -> Result<OwnedTensorPtr, CoreError> {
    let host = input.read_to_cpu()?;
    let dims: Vec<usize> = host.shape().dims().to_vec();
    match host.dtype() {
        DataType::F32 => {
            let data = bytes_as_vec_f32(host.as_bytes(), host.shape().element_count())?;
            let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&dims), data)
                .map_err(|e| CoreError::InferenceFailed(e.to_string()))?;
            Ok(OwnedTensorPtr::F32(
                TensorPtr::from_array(arr).map_err(|e| {
                    CoreError::InferenceFailed(format!("TensorPtr::from_array failed: {:?}", e))
                })?,
            ))
        }
        DataType::U8 => {
            let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&dims), host.as_bytes().to_vec())
                .map_err(|e| CoreError::InferenceFailed(e.to_string()))?;
            Ok(OwnedTensorPtr::U8(
                TensorPtr::from_array(arr).map_err(|e| {
                    CoreError::InferenceFailed(format!("TensorPtr::from_array failed: {:?}", e))
                })?,
            ))
        }
        DataType::I32 => {
            let data = bytes_as_vec_i32(host.as_bytes(), host.shape().element_count())?;
            let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&dims), data)
                .map_err(|e| CoreError::InferenceFailed(e.to_string()))?;
            Ok(OwnedTensorPtr::I32(
                TensorPtr::from_array(arr).map_err(|e| {
                    CoreError::InferenceFailed(format!("TensorPtr::from_array failed: {:?}", e))
                })?,
            ))
        }
        DataType::I64 => {
            let data = bytes_as_vec_i64(host.as_bytes(), host.shape().element_count())?;
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

pub fn evalue_to_tensor_buffer(
    value: &EValue<'_>,
    device: &Device,
    desc: &crate::program::TensorDescriptor,
) -> Result<ExecuTorchTensorBuffer, ExecuTorchError> {
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

    ExecuTorchTensorBuffer::new(device.clone(), shape, dtype, bytes)
}
