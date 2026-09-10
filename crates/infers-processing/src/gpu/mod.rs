mod build;
mod dispatch;
mod processor;
mod resources;
mod staging;
mod target;
mod util;

pub(crate) const SHADERS: &[u8] = include_bytes!(env!("PROCESSING_SHADERS_SPV"));

pub use processor::GpuImageProcessor;
pub use staging::{set_session_staging_query, SessionStagingQuery};
