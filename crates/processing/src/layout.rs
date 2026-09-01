use infers_core::{CoreError, Cpu, DataType, Tensor, TensorShape};
use processing_core::TensorLayout;

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
pub fn nhwc_to_nchw_cpu(input: &Tensor<Cpu>) -> Result<Tensor<Cpu>, CoreError> {
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

    let host = input.read_to_host()?;
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
    let out_host = infers_core::HostTensor::from_f32(shape, nchw)?;
    Tensor::from_host(&Cpu, &out_host)
}

/// Convert layout on CPU tensors.
pub fn convert_layout_cpu(
    input: &Tensor<Cpu>,
    target_layout: TensorLayout,
) -> Result<Tensor<Cpu>, CoreError> {
    let current = infer_rgb_layout(input.shape())?;
    if current == target_layout {
        return Ok(input.clone());
    }
    match (current, target_layout) {
        (TensorLayout::Nhwc, TensorLayout::Nchw) => nhwc_to_nchw_cpu(input),
        (TensorLayout::Nchw, TensorLayout::Nhwc) => nchw_to_nhwc_cpu(input),
        _ => Err(CoreError::InvalidShape("unsupported layout conversion".into())),
    }
}

fn nchw_to_nhwc_cpu(input: &Tensor<Cpu>) -> Result<Tensor<Cpu>, CoreError> {
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
    let host = input.read_to_host()?;
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
    let shape = TensorShape::new([1, h, w, 3])?;
    let out_host = infers_core::HostTensor::from_f32(shape, nhwc)?;
    Tensor::from_host(&Cpu, &out_host)
}

#[cfg(feature = "vulkan")]
/// Convert layout on GPU tensors (Vulkan device only).
pub fn convert_layout_vulkan(
    input: &Tensor<infers_gpu::Vulkan>,
    target_layout: TensorLayout,
) -> Result<Tensor<infers_gpu::Vulkan>, CoreError> {
    let current = infer_rgb_layout(input.shape())?;
    if current == target_layout {
        return Ok(input.clone());
    }
    match (current, target_layout) {
        (TensorLayout::Nhwc, TensorLayout::Nchw) => {
            crate::gpu::nhwc_to_nchw_gpu(input.device(), input)
        }
        _ => Err(CoreError::InvalidShape(
            "GPU layout conversion supports NHWC→NCHW only".into(),
        )),
    }
}
