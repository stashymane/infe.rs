pub use infers_processing_core::TensorLayout;

use crate::error::CoreError;
use crate::tensor::TensorShape;

/// `[N, C, H, W]` shape for the given extents.
pub fn shape_nchw(n: usize, c: usize, h: usize, w: usize) -> Result<TensorShape, CoreError> {
    TensorShape::new([n, c, h, w])
}

/// `[N, H, W, C]` shape for the given extents.
pub fn shape_nhwc(n: usize, c: usize, h: usize, w: usize) -> Result<TensorShape, CoreError> {
    TensorShape::new([n, h, w, c])
}

/// Batch-major RGB shape for `layout` with batch `n`, `c` channels, and spatial `h`×`w`.
pub fn shape_for(
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    layout: TensorLayout,
) -> Result<TensorShape, CoreError> {
    match layout {
        TensorLayout::Nhwc => shape_nhwc(n, c, h, w),
        TensorLayout::Nchw => shape_nchw(n, c, h, w),
    }
}
