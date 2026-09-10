use crate::error::ExecuTorchError;
use crate::program::TensorDescriptor;
use crate::tensor::{scalar_type_to_data_type};
use executorch::evalue::{EValue, IntoEValue, Tag};
use executorch::tensor::{TensorPtr, TensorPtrBuilder, View};
use infers_core::{cast_bytes, CoreError, Cpu, DataType, HostTensor, Tensor, TensorShape};

pub enum OwnedTensorPtr {
    F32(TensorPtr<'static, View<f32>>),
    U8(TensorPtr<'static, View<u8>>),
    I32(TensorPtr<'static, View<i32>>),
    I64(TensorPtr<'static, View<i64>>),
}

impl OwnedTensorPtr {
    pub fn as_evalue(&self) -> executorch::evalue::EValue<'_> {
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
    pub fn as_evalue(&self) -> executorch::evalue::EValue<'_> {
        match self {
            Self::F32(p) => p.into_evalue(),
            Self::U8(p) => p.into_evalue(),
            Self::I32(p) => p.into_evalue(),
            Self::I64(p) => p.into_evalue(),
        }
    }
}

pub struct HostInputPlan<'a> {
    pub tensor_ptrs: Vec<HostTensorPtr<'a>>,
}

impl Drop for HostInputPlan<'_> {
    fn drop(&mut self) {
        self.tensor_ptrs.clear();
    }
}

impl<'a> HostInputPlan<'a> {
    pub fn build(inputs: &[&'a Tensor<Cpu>]) -> Result<Self, CoreError> {
        let mut plan = Self {
            tensor_ptrs: Vec::with_capacity(inputs.len()),
        };
        for input in inputs {
            plan.push_input(input)?;
        }
        Ok(plan)
    }

    fn push_input(&mut self, input: &'a Tensor<Cpu>) -> Result<(), CoreError> {
        let shape = input.shape();
        let dims = dims_i32(shape);
        let bytes = &input.storage().0;
        match input.dtype() {
            DataType::F32 => {
                let slice = cast_bytes(bytes)?;
                self.tensor_ptrs.push(build_f32_view(slice, shape, &dims)?);
            }
            DataType::U8 => {
                self.tensor_ptrs.push(build_u8_view(bytes, shape, &dims)?);
            }
            DataType::I32 => {
                let slice = cast_bytes(bytes)?;
                self.tensor_ptrs.push(build_i32_view(slice, shape, &dims)?);
            }
            DataType::I64 => {
                let slice = cast_bytes(bytes)?;
                self.tensor_ptrs.push(build_i64_view(slice, shape, &dims)?);
            }
            other => {
                return Err(CoreError::InvalidDataType {
                    expected: DataType::F32,
                    actual: other,
                });
            }
        }
        Ok(())
    }
}

fn dims_i32(shape: &TensorShape) -> Vec<i32> {
    shape.dims().iter().map(|&d| d as i32).collect()
}

fn validate_element_count<T>(slice: &[T], shape: &TensorShape) -> Result<(), CoreError> {
    if slice.len() != shape.element_count() {
        return Err(CoreError::InvalidShape(format!(
            "Element count mismatch: shape requires {}, got {}",
            shape.element_count(),
            slice.len()
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
    let ptr = unsafe {
        TensorPtrBuilder::<View<i64>>::from_ptr(slice.as_ptr(), dims.iter().copied())
            .build()
            .map_err(|e| CoreError::InferenceFailed(format!("TensorPtr build failed: {e:?}")))?
    };
    Ok(HostTensorPtr::I64(ptr))
}

pub fn evalue_to_cpu_tensor(
    value: &EValue<'_>,
    desc: &TensorDescriptor,
) -> Result<Tensor<Cpu>, ExecuTorchError> {
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
    let host = HostTensor::new(shape, dtype, bytes)?;
    Ok(Tensor::adopt_host(&host))
}
