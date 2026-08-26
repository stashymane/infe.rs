#![feature(variant_count)]

#[macro_use]
mod mirror;
pub mod backend;
pub mod device;
pub mod error;
#[cfg(feature = "vulkan")]
pub mod gpu_context;
pub mod image;
pub mod session;
pub mod tensor;

#[cfg(target_os = "android")]
mod android_jni;

pub use backend::*;
pub use device::*;
pub use error::*;
#[cfg(feature = "vulkan")]
pub use gpu_context::*;
pub use image::*;
pub use session::*;
pub use tensor::*;

uniffi::setup_scaffolding!();
