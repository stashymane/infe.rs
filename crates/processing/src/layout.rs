use infers_core::{
    CoreError, CpuTensor, DataType, Device, DeviceTransfer, TensorBuffer, TensorShape,
};
use processing_core::TensorLayout;

#[cfg(feature = "vulkan")]
use crate::gpu::buffer::GpuTensorBuffer;

/// Infer channel layout from a 4-D RGB tensor shape.
pub fn infer_rgb_layout(shape: &TensorShape) -> Result<TensorLayout, CoreError> {
    let dims = shape.dims();
    if dims.len() != 4 || dims[0] != 1 {
        return Err(CoreError::InvalidShape(format!(
            "expected 4-D batch-1 RGB shape, got {dims:?}"
        )));
    }
    if dims[3] == 3 {
        Ok(TensorLayout::Nhwc)
    } else if dims[1] == 3 {
        Ok(TensorLayout::Nchw)
    } else {
        Err(CoreError::InvalidShape(format!(
            "cannot infer NHWC/NCHW from shape {dims:?}"
        )))
    }
}

/// Convert `[1, H, W, 3]` NHWC f32 to `[1, 3, H, W]` NCHW on the host.
pub fn nhwc_to_nchw_cpu(input: &dyn TensorBuffer) -> Result<CpuTensor<f32>, CoreError> {
    let dims = input.shape().dims();
    if infer_rgb_layout(input.shape())? != TensorLayout::Nhwc {
        return Err(CoreError::InvalidShape(format!(
            "expected NHWC [1, H, W, 3], got {dims:?}"
        )));
    }
    if input.dtype() != DataType::F32 {
        return Err(CoreError::InvalidDataType {
            expected: DataType::F32,
            actual: input.dtype(),
        });
    }

    let host = input.read_to_cpu()?;
    let src = host.as_slice_f32()?;
    let h = dims[1];
    let w = dims[2];
    let hw = h * w;

    let mut nchw = vec![0.0f32; 3 * hw];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let base = i * 3;
            nchw[i] = src[base];
            nchw[hw + i] = src[base + 1];
            nchw[2 * hw + i] = src[base + 2];
        }
    }

    let shape = TensorShape::new([1, 3, h, w])?;
    CpuTensor::from_f32(shape, nchw)
}

/// Convert layout, uploading to GPU when `transfer` is provided and the input is host-resident.
pub fn convert_layout(
    input: &dyn TensorBuffer,
    target_layout: TensorLayout,
    transfer: Option<&dyn DeviceTransfer>,
) -> Result<Box<dyn TensorBuffer>, CoreError> {
    let current = infer_rgb_layout(input.shape())?;
    if current == target_layout {
        return match input.device().is_cpu() {
            true => {
                let host = input.read_to_cpu()?;
                Ok(host_to_tensor_buffer(host.as_ref(), input.shape(), input.dtype())?)
            }
            false => {
                #[cfg(feature = "vulkan")]
                if let Some(gpu) = input.as_any().downcast_ref::<GpuTensorBuffer>() {
                    return Ok(Box::new(gpu.clone()));
                }
                Err(CoreError::BufferTransferFailed(
                    "unsupported device-resident tensor for layout pass-through".into(),
                ))
            }
        };
    }

    match (current, target_layout) {
        (TensorLayout::Nhwc, TensorLayout::Nchw) => {
            #[cfg(feature = "vulkan")]
            if let Some(gpu) = input.as_any().downcast_ref::<GpuTensorBuffer>() {
                let out = crate::gpu::layout_pass::nhwc_to_nchw_gpu(gpu.vulkan_context(), gpu)?;
                return Ok(Box::new(out));
            }
            let cpu = nhwc_to_nchw_cpu(input)?;
            if let Some(transfer) = transfer {
                return transfer.upload_tensor(&cpu);
            }
            Ok(Box::new(cpu))
        }
        (TensorLayout::Nchw, TensorLayout::Nhwc) => {
            let cpu = nchw_to_nhwc_cpu(input)?;
            if let Some(transfer) = transfer {
                return transfer.upload_tensor(&cpu);
            }
            Ok(Box::new(cpu))
        }
        _ => Err(CoreError::InvalidShape("unsupported layout conversion".into())),
    }
}

fn nchw_to_nhwc_cpu(input: &dyn TensorBuffer) -> Result<CpuTensor<f32>, CoreError> {
    let dims = input.shape().dims();
    if infer_rgb_layout(input.shape())? != TensorLayout::Nchw {
        return Err(CoreError::InvalidShape(format!(
            "expected NCHW [1, 3, H, W], got {dims:?}"
        )));
    }
    if input.dtype() != DataType::F32 {
        return Err(CoreError::InvalidDataType {
            expected: DataType::F32,
            actual: input.dtype(),
        });
    }
    let host = input.read_to_cpu()?;
    let src = host.as_slice_f32()?;
    let h = dims[2];
    let w = dims[3];
    let hw = h * w;
    let mut nhwc = vec![0.0f32; 3 * hw];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            nhwc[i * 3] = src[i];
            nhwc[i * 3 + 1] = src[hw + i];
            nhwc[i * 3 + 2] = src[2 * hw + i];
        }
    }
    CpuTensor::from_f32(TensorShape::new([1, h, w, 3])?, nhwc)
}

fn host_to_tensor_buffer(
    host: &dyn infers_core::AnyHostTensor,
    shape: &TensorShape,
    dtype: DataType,
) -> Result<Box<dyn TensorBuffer>, CoreError> {
    match dtype {
        DataType::F32 => Ok(Box::new(CpuTensor::from_f32(
            shape.clone(),
            host.as_slice_f32()?.to_vec(),
        )?)),
        DataType::U8 => Ok(Box::new(CpuTensor::from_u8(
            shape.clone(),
            host.as_slice_u8()?.to_vec(),
        )?)),
        other => Err(CoreError::InvalidDataType {
            expected: DataType::F32,
            actual: other,
        }),
    }
}

/// Prepare a tensor for model inference: convert layout if needed and upload to the session device.
pub fn model_input_from_preprocess(
    tensor: &dyn TensorBuffer,
    expected_shape: &TensorShape,
    session_device: &Device,
    transfer: Option<&dyn DeviceTransfer>,
) -> Result<Box<dyn TensorBuffer>, CoreError> {
    let target_layout = infer_rgb_layout(expected_shape)?;
    let mut current: Box<dyn TensorBuffer> = if tensor.shape() == expected_shape {
        #[cfg(feature = "vulkan")]
        if !session_device.is_cpu() {
            if let Some(gpu) = tensor.as_any().downcast_ref::<GpuTensorBuffer>() {
                if gpu.device().kind == session_device.kind {
                    return Ok(Box::new(gpu.clone()));
                }
            }
        }
        host_to_tensor_buffer(
            tensor.read_to_cpu()?.as_ref(),
            tensor.shape(),
            tensor.dtype(),
        )?
    } else {
        convert_layout(tensor, target_layout, None)?
    };

    if current.shape() != expected_shape {
        current = convert_layout(current.as_ref(), target_layout, None)?;
    }

    if current.device().kind != session_device.kind {
        current = current.copy_to_device(session_device, transfer)?;
    }
    Ok(current)
}
