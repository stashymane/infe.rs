//! Compile-fail: host image cannot be passed to GPU image processor.

#[cfg(feature = "vulkan")]
#[test]
fn ui() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
}
