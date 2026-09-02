pub mod processor;

pub(crate) const SHADERS: &[u8] = include_bytes!(env!("PROCESSING_SHADERS_SPV"));

pub use processor::{
    set_session_staging_query, GpuImageProcessor, GpuImageProcessorOptions, SessionStagingQuery,
};
