use crate::error::ExecuTorchError;
use crate::program::TensorDescriptor;
use crate::tensor::{scalar_type_to_data_type, ExecuTorchTensorBuffer};
use executorch::evalue::{EValue, IntoEValue, Tag};
use executorch::tensor::{Scalar, TensorPtr, TensorPtrBuilder, View};
use infers_core::{
    bytes_to_vec, cast_bytes, CoreError, CpuTensor, DataType, Device, TensorBuffer, TensorShape,
};

pub enum OwnedTensorPtr {
    F32(TensorPtr<'static, View<f32>>),
    U8(TensorPtr<'static, View<u8>>),
    I32(TensorPtr<'static, View<i32>>),
    I64(TensorPtr<'static, View<i64>>),
}

impl OwnedTensorPtr {
    pub fn as_evalue(&self) -> EValue<'_> {
        match self {
            Self::F32(p) => p.into_evalue(),
            Self::U8(p) => p.into_evalue(),
            Self::I32(p) => p.into_evalue(),
            Self::I64(p) => p.into_evalue(),
        }
    }
}

pub enum HostTensorPtr<'a> {
    F32(TensorPtr<'a, View<f32>>),
    U8(TensorPtr<'a, View<u8>>),
    I32(TensorPtr<'a, View<i32>>),
    I64(TensorPtr<'a, View<i64>>),
}

impl HostTensorPtr<'_> {
    pub fn as_evalue(&self) -> EValue<'_> {
        match self {
            Self::F32(p) => p.into_evalue(),
            Self::U8(p) => p.into_evalue(),
            Self::I32(p) => p.into_evalue(),
            Self::I64(p) => p.into_evalue(),
        }
    }
}

/// Pins borrowed host tensor views for one `execute` call.
///
/// When an input cannot be borrowed, `from_vec` moves a single host copy into
/// the ExecuTorch tensor (no separate backing field). Drop order matches
/// [`crate::gpu_input::GpuInputPlan`]: clear tensor pointers first.
pub struct HostInputPlan<'a> {
    pub tensor_ptrs: Vec<HostTensorPtr<'a>>,
}

impl Drop for HostInputPlan<'_> {
    fn drop(&mut self) {
        self.tensor_ptrs.clear();
    }
}

impl<'a> HostInputPlan<'a> {
    pub fn build(inputs: &[&'a dyn TensorBuffer]) -> Result<Self, CoreError> {
        let mut plan = Self {
            tensor_ptrs: Vec::with_capacity(inputs.len()),
        };
        for &input in inputs {
            plan.push_input(input)?;
        }
        Ok(plan)
    }

    fn push_input(&mut self, input: &'a dyn TensorBuffer) -> Result<(), CoreError> {
        if let Some(ptr) = try_borrowed_ptr(input)? {
            self.tensor_ptrs.push(ptr);
            return Ok(());
        }
        self.tensor_ptrs.push(owned_ptr_from_readback(input)?);
        Ok(())
    }
}

#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
pub fn tensor_ptr_from_host(input: &dyn TensorBuffer) -> Result<OwnedTensorPtr, CoreError> {
    if let Some(ptr) = try_borrowed_ptr(input)? {
        return host_ptr_to_owned(ptr);
    }
    let host = input.read_to_cpu()?;
    owned_ptr_from_any_host(host.as_ref())
}

fn dims_i32(shape: &TensorShape) -> Vec<i32> {
    shape.dims().iter().map(|&d| d as i32).collect()
}

fn try_borrowed_ptr(input: &dyn TensorBuffer) -> Result<Option<HostTensorPtr<'_>>, CoreError> {
    let shape = input.shape();
    let dims = dims_i32(shape);
    let ptr = match input.dtype() {
        DataType::F32 => {
            if let Some(t) = input.as_any().downcast_ref::<CpuTensor<f32>>() {
                Some(build_f32_view(t.as_slice(), shape, &dims)?)
            } else if let Some(t) = input.as_any().downcast_ref::<ExecuTorchTensorBuffer>() {
                if t.dtype() == DataType::F32 {
                    Some(build_f32_view(
                        t.as_slice_f32().map_err(CoreError::from)?,
                        shape,
                        &dims,
                    )?)
                } else {
                    None
                }
            } else {
                None
            }
        }
        DataType::U8 => {
            if let Some(t) = input.as_any().downcast_ref::<CpuTensor<u8>>() {
                Some(build_u8_view(t.as_slice(), shape, &dims)?)
            } else if let Some(t) = input.as_any().downcast_ref::<ExecuTorchTensorBuffer>() {
                if t.dtype() == DataType::U8 {
                    Some(build_u8_view(t.as_bytes(), shape, &dims)?)
                } else {
                    None
                }
            } else {
                None
            }
        }
        DataType::I32 => {
            if let Some(t) = input.as_any().downcast_ref::<CpuTensor<i32>>() {
                Some(build_i32_view(t.as_slice(), shape, &dims)?)
            } else if let Some(t) = input.as_any().downcast_ref::<ExecuTorchTensorBuffer>() {
                if t.dtype() == DataType::I32 {
                    let need = shape.byte_size(DataType::I32);
                    let slice = cast_bytes::<i32>(&t.as_bytes()[..need])?;
                    Some(build_i32_view(slice, shape, &dims)?)
                } else {
                    None
                }
            } else {
                None
            }
        }
        DataType::I64 => {
            if let Some(t) = input.as_any().downcast_ref::<CpuTensor<i64>>() {
                Some(build_i64_view(t.as_slice(), shape, &dims)?)
            } else if let Some(t) = input.as_any().downcast_ref::<ExecuTorchTensorBuffer>() {
                if t.dtype() == DataType::I64 {
                    let need = shape.byte_size(DataType::I64);
                    let slice = cast_bytes::<i64>(&t.as_bytes()[..need])?;
                    Some(build_i64_view(slice, shape, &dims)?)
                } else {
                    None
                }
            } else {
                None
            }
        }
        _ => None,
    };
    Ok(ptr)
}

fn owned_ptr_from_readback(input: &dyn TensorBuffer) -> Result<HostTensorPtr<'static>, CoreError> {
    let host = input.read_to_cpu()?;
    let owned = owned_ptr_from_any_host(host.as_ref())?;
    Ok(host_owned_to_static(owned))
}

fn owned_ptr_from_any_host(host: &dyn infers_core::AnyHostTensor) -> Result<OwnedTensorPtr, CoreError> {
    let shape = host.shape();
    let dims = dims_i32(shape);
    match host.dtype() {
        DataType::F32 => {
            let need = shape.byte_size(DataType::F32);
            let data = bytes_to_vec::<f32>(&host.as_bytes()[..need])?;
            Ok(OwnedTensorPtr::F32(build_f32_from_vec(data, &dims)?))
        }
        DataType::U8 => {
            let data = {
                let mut data = Vec::with_capacity(host.as_bytes().len());
                data.extend_from_slice(host.as_bytes());
                data
            };
            Ok(OwnedTensorPtr::U8(build_u8_from_vec(data, &dims)?))
        }
        DataType::I32 => {
            let need = shape.byte_size(DataType::I32);
            let data = bytes_to_vec::<i32>(&host.as_bytes()[..need])?;
            Ok(OwnedTensorPtr::I32(build_i32_from_vec(data, &dims)?))
        }
        DataType::I64 => {
            let need = shape.byte_size(DataType::I64);
            let data = bytes_to_vec::<i64>(&host.as_bytes()[..need])?;
            Ok(OwnedTensorPtr::I64(build_i64_from_vec(data, &dims)?))
        }
        other => Err(CoreError::InferenceFailed(format!(
            "Unsupported input dtype for ExecuTorch Module: {:?}",
            other
        ))),
    }
}

fn validate_element_count<T>(slice: &[T], shape: &TensorShape) -> Result<(), CoreError> {
    if slice.len() < shape.element_count() {
        return Err(CoreError::BufferTransferFailed(format!(
            "input has {} elements, shape requires {}",
            slice.len(),
            shape.element_count()
        )));
    }
    Ok(())
}

fn build_f32_view<'a>(
    slice: &'a [f32],
    shape: &TensorShape,
    dims: &[i32],
) -> Result<HostTensorPtr<'a>, CoreError> {
    validate_element_count(slice, shape)?;
    if !slice.as_ptr().is_aligned() {
        return Err(CoreError::BufferTransferFailed(
            "f32 input slice is not aligned".into(),
        ));
    }
    // SAFETY: `slice` covers the full static shape (checked above) and remains
    // valid through `execute` because the source `TensorBuffer` outlives
    // [`HostInputPlan`].
    let ptr = unsafe {
        TensorPtrBuilder::<View<f32>>::from_ptr(slice.as_ptr(), dims.iter().copied())
            .build()
            .map_err(|e| CoreError::InferenceFailed(format!("TensorPtr build failed: {e:?}")))?
    };
    Ok(HostTensorPtr::F32(ptr))
}

fn build_u8_view<'a>(
    slice: &'a [u8],
    shape: &TensorShape,
    dims: &[i32],
) -> Result<HostTensorPtr<'a>, CoreError> {
    validate_element_count(slice, shape)?;
    // SAFETY: as for `build_f32_view`; `u8` has no alignment requirement.
    let ptr = unsafe {
        TensorPtrBuilder::<View<u8>>::from_ptr(slice.as_ptr(), dims.iter().copied())
            .build()
            .map_err(|e| CoreError::InferenceFailed(format!("TensorPtr build failed: {e:?}")))?
    };
    Ok(HostTensorPtr::U8(ptr))
}

fn build_i32_view<'a>(
    slice: &'a [i32],
    shape: &TensorShape,
    dims: &[i32],
) -> Result<HostTensorPtr<'a>, CoreError> {
    validate_element_count(slice, shape)?;
    if !slice.as_ptr().is_aligned() {
        return Err(CoreError::BufferTransferFailed(
            "i32 input slice is not aligned".into(),
        ));
    }
    // SAFETY: as for `build_f32_view`.
    let ptr = unsafe {
        TensorPtrBuilder::<View<i32>>::from_ptr(slice.as_ptr(), dims.iter().copied())
            .build()
            .map_err(|e| CoreError::InferenceFailed(format!("TensorPtr build failed: {e:?}")))?
    };
    Ok(HostTensorPtr::I32(ptr))
}

fn build_i64_view<'a>(
    slice: &'a [i64],
    shape: &TensorShape,
    dims: &[i32],
) -> Result<HostTensorPtr<'a>, CoreError> {
    validate_element_count(slice, shape)?;
    if !slice.as_ptr().is_aligned() {
        return Err(CoreError::BufferTransferFailed(
            "i64 input slice is not aligned".into(),
        ));
    }
    // SAFETY: as for `build_f32_view`.
    let ptr = unsafe {
        TensorPtrBuilder::<View<i64>>::from_ptr(slice.as_ptr(), dims.iter().copied())
            .build()
            .map_err(|e| CoreError::InferenceFailed(format!("TensorPtr build failed: {e:?}")))?
    };
    Ok(HostTensorPtr::I64(ptr))
}

fn build_f32_from_vec(data: Vec<f32>, dims: &[i32]) -> Result<TensorPtr<'static, View<f32>>, CoreError> {
    // SAFETY: `dims` matches the moved `data` length (validated at readback).
    let ptr = unsafe {
        TensorPtrBuilder::<View<f32>>::from_vec(data)
            .sizes(dims.iter().copied())
            .build()
            .map_err(|e| CoreError::InferenceFailed(format!("TensorPtr build failed: {e:?}")))?
    };
    Ok(ptr)
}

fn build_u8_from_vec(data: Vec<u8>, dims: &[i32]) -> Result<TensorPtr<'static, View<u8>>, CoreError> {
    let ptr = unsafe {
        TensorPtrBuilder::<View<u8>>::from_vec(data)
            .sizes(dims.iter().copied())
            .build()
            .map_err(|e| CoreError::InferenceFailed(format!("TensorPtr build failed: {e:?}")))?
    };
    Ok(ptr)
}

fn build_i32_from_vec(data: Vec<i32>, dims: &[i32]) -> Result<TensorPtr<'static, View<i32>>, CoreError> {
    let ptr = unsafe {
        TensorPtrBuilder::<View<i32>>::from_vec(data)
            .sizes(dims.iter().copied())
            .build()
            .map_err(|e| CoreError::InferenceFailed(format!("TensorPtr build failed: {e:?}")))?
    };
    Ok(ptr)
}

fn build_i64_from_vec(data: Vec<i64>, dims: &[i32]) -> Result<TensorPtr<'static, View<i64>>, CoreError> {
    let ptr = unsafe {
        TensorPtrBuilder::<View<i64>>::from_vec(data)
            .sizes(dims.iter().copied())
            .build()
            .map_err(|e| CoreError::InferenceFailed(format!("TensorPtr build failed: {e:?}")))?
    };
    Ok(ptr)
}

#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
fn host_ptr_to_owned<'a>(view: HostTensorPtr<'a>) -> Result<OwnedTensorPtr, CoreError> {
    Ok(match view {
        HostTensorPtr::F32(p) => OwnedTensorPtr::F32(erasure_static(p)),
        HostTensorPtr::U8(p) => OwnedTensorPtr::U8(erasure_static(p)),
        HostTensorPtr::I32(p) => OwnedTensorPtr::I32(erasure_static(p)),
        HostTensorPtr::I64(p) => OwnedTensorPtr::I64(erasure_static(p)),
    })
}

fn host_owned_to_static(ptr: OwnedTensorPtr) -> HostTensorPtr<'static> {
    match ptr {
        OwnedTensorPtr::F32(p) => HostTensorPtr::F32(p),
        OwnedTensorPtr::U8(p) => HostTensorPtr::U8(p),
        OwnedTensorPtr::I32(p) => HostTensorPtr::I32(p),
        OwnedTensorPtr::I64(p) => HostTensorPtr::I64(p),
    }
}

#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
fn erasure_static<'a, S: Scalar>(ptr: TensorPtr<'a, View<S>>) -> TensorPtr<'static, View<S>> {
    // SAFETY: caller guarantees the borrow outlives `execute`; same obligation
    // as [`crate::gpu_input::HostVisibleInput::tensor_ptr`].
    unsafe { std::mem::transmute(ptr) }
}

pub fn evalue_to_tensor_buffer(
    value: &EValue<'_>,
    device: &Device,
    desc: &TensorDescriptor,
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
        // SAFETY: `ptr` is non-null (checked above) and ExecuTorch reports
        // `nbytes` as the readable length of this output tensor's data, which
        // stays valid until the next `execute` on the module. The copy happens
        // before returning, so the borrow does not escape.
        unsafe { std::slice::from_raw_parts(ptr, nbytes).to_vec() }
    };

    let sizes = tensor.sizes();
    let shape = TensorShape::new(sizes.iter().map(|&d| d as usize).collect::<Vec<_>>())?;
    let dtype = scalar_type_to_data_type(tensor.scalar_type())?;

    ExecuTorchTensorBuffer::new(device.clone(), shape, dtype, bytes)
}
