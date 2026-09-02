use processing_core::ProcessingOptions;

#[test]
fn processing_options_matches_shader_push_constant_layout() {
    assert_eq!(core::mem::size_of::<ProcessingOptions>(), 52);
}
