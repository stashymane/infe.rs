//! Core types for on-device ML inference: tensors, devices, image buffers, and backend traits.

pub mod backend;
pub mod bytes;
pub mod device;
pub mod error;
pub mod image;
pub mod tensor;

pub use backend::{Backend, ModelSession, SessionConfig};
pub use bytes::{bytes_to_vec, cast_bytes};
pub use device::{Device, DeviceKind};
pub use error::CoreError;
pub use image::{
    CpuImageBuffer, FitMode, ImageFormat, ImageInputBuffer, ProcessingOptions, Rotation,
};
pub use tensor::{AnyHostTensor, CpuTensor, DataType, TensorBuffer, TensorShape};
