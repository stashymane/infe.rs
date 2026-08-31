use processing_core::ProcessingOptions;

#[test]
fn processing_options_matches_shader_uniform_layout() {
    assert_eq!(core::mem::size_of::<ProcessingOptions>(), 52);
}
