use crate::error::CoreError;
use std::sync::atomic::{AtomicBool, Ordering};

/// Runtime guard against double-commit when move semantics are not enforced (e.g. UniFFI).
#[derive(Debug, Default)]
pub struct ConsumedFlag(AtomicBool);

impl ConsumedFlag {
    pub fn take(&self) -> Result<(), CoreError> {
        if self.0.swap(true, Ordering::AcqRel) {
            Err(CoreError::AlreadyConsumed)
        } else {
            Ok(())
        }
    }
}
