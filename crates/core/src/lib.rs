//! Core types for on-device ML inference: tensors, devices, image buffers, and backend traits.

pub mod backend;
pub mod bytes;
pub mod device;
pub mod error;
pub mod image;
pub mod layout;
pub mod tensor;
pub mod transfer;

pub use backend::{Backend, ModelSession, SessionConfig};
pub use bytes::{bytes_to_vec, cast_bytes};
pub use device::{Device, DeviceKind};
pub use error::CoreError;
pub use image::{
    CpuImageBuffer, FitMode, ImageFormat, ImageInputBuffer, ProcessingOptions, Rotation,
    TensorLayout,
};
pub use layout::{shape_for, shape_nchw, shape_nhwc};
pub use tensor::{AnyHostTensor, CpuTensor, DataType, TensorBuffer, TensorShape};
pub use transfer::DeviceTransfer;
