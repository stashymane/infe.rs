/// Restricts [`Device`](crate::Device) to in-tree implementors.
///
/// # Safety
///
/// Implementing this outside the `infers` workspace is unsupported: the device
/// invariants `Tensor<D>` relies on are upheld by in-tree implementors only.
#[doc(hidden)]
pub unsafe trait Sealed {}

unsafe impl Sealed for crate::Cpu {}
