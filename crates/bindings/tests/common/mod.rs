use std::sync::Arc;

use infers_bindings::{Device, ModelSession};
use infers_core::Backend;
use infers_test_utils::MockBackend;

pub fn load_mock_session(model_bytes: Vec<u8>, device: Device) -> Arc<ModelSession> {
    let backend = MockBackend::new("mock_backend");
    let core_device: infers_core::Device = device.into();
    let session = backend
        .load_model(&model_bytes, &core_device)
        .expect("Load model on mock backend failed");
    Arc::new(ModelSession::new(session))
}
