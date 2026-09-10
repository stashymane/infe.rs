use infers_core::CoreError;
use infers_gpu::VulkanBufferHandle;
use std::sync::Arc;

/// Query ExecuTorch Vulkan input staging buffers by session input slot.
pub type SessionStagingQuery =
    Arc<dyn Fn(usize) -> Result<VulkanBufferHandle, CoreError> + Send + Sync>;

thread_local! {
    static SESSION_STAGING: std::cell::RefCell<Option<SessionStagingQuery>> =
        const { std::cell::RefCell::new(None) };
}

/// Install a thread-local staging query for [`MaterializeTarget::SessionInput`] commits.
pub fn set_session_staging_query(query: Option<SessionStagingQuery>) {
    SESSION_STAGING.with(|slot| *slot.borrow_mut() = query);
}

pub(crate) fn staging_for_materialize(
    explicit: Option<&SessionStagingQuery>,
) -> Option<SessionStagingQuery> {
    explicit
        .cloned()
        .or_else(|| SESSION_STAGING.with(|slot| slot.borrow().clone()))
}
