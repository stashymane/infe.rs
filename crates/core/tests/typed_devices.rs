//! Compile-fail tests for typed device invariants.

#[cfg(all(feature = "vulkan", not(target_os = "android")))]
#[test]
fn ui() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
}
