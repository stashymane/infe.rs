//! Core types for on-device ML inference: tensors, devices, image buffers, and backend traits.

pub mod backend;
pub mod bytes;
pub mod consumed;
pub mod deferred;
pub mod device;
pub mod error;
pub mod image;
pub mod infer_input;
pub mod layout;
pub mod pending;
pub mod sealed;
pub mod tensor;

pub use backend::{Session, SessionInputSink};
pub use bytes::{bytes_to_vec, cast_bytes, vec_to_bytes};
pub use consumed::ConsumedFlag;
pub use deferred::Deferred;
pub use device::{Cpu, Device, DeviceInfo, DeviceKind};
pub use error::CoreError;
pub use image::{
    CpuImage, FitMode, HardwareImage, Image, ImageFormat, ProcessingOptions, TensorLayout,
};
pub use infer_input::{prepare_infer, InferInput};
pub use layout::{shape_for, shape_nchw, shape_nhwc};
pub use pending::{MaterializeTarget, Pending};
pub use tensor::{
    DataType, HostBytes, HostTensor, Tensor, TensorAdopt, TensorShape,
};
