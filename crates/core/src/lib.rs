//! Core types for on-device ML inference: tensors, devices, image buffers, and backend traits.

pub mod backend;
pub mod bytes;
pub mod device;
pub mod error;
pub mod image;
pub mod layout;
pub mod sealed;
pub mod tensor;

pub use backend::Session;
pub use bytes::{bytes_to_vec, cast_bytes, vec_to_bytes};
pub use device::{Cpu, Device, DeviceInfo, DeviceKind};
pub use error::CoreError;
pub use image::{
    DeviceImage, FitMode, HostImage, ImageFormat, ProcessingOptions, Rotation, TensorLayout,
};
pub use layout::{shape_for, shape_nchw, shape_nhwc};
pub use tensor::{
    DataType, HostBytes, HostTensor, Tensor, TensorAdopt, TensorShape,
};
