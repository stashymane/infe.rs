//! Compile-fail: host image cannot be passed to GPU image processor.

#[cfg(all(feature = "vulkan", not(target_os = "android")))]
#[test]
fn ui() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
}
