//! Compile-fail tests for typed device invariants.

#[cfg(feature = "vulkan")]
#[test]
fn ui() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
}
