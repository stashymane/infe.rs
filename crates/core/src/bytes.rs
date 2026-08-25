use bytemuck::Pod;

use crate::error::CoreError;

#[inline]
pub fn cast_bytes<T: Pod>(bytes: &[u8]) -> Result<&[T], CoreError> {
    let elem_size = std::mem::size_of::<T>();
    if elem_size == 0 {
        return Err(CoreError::InvalidArgument(format!(
            "cannot cast bytes to zero-sized type {}",
            std::any::type_name::<T>()
        )));
    }
    if !bytes.len().is_multiple_of(elem_size) {
        return Err(CoreError::InvalidArgument(format!(
            "byte length {} is not a multiple of element size {}",
            bytes.len(),
            elem_size
        )));
    }
    bytemuck::try_cast_slice(bytes).map_err(|_| {
        CoreError::BufferTransferFailed(format!(
            "failed to cast byte slice to {}",
            std::any::type_name::<T>()
        ))
    })
}

#[inline]
pub fn bytes_to_vec<T: Pod>(bytes: &[u8]) -> Result<Vec<T>, CoreError> {
    cast_bytes(bytes).map(|slice| slice.to_vec())
}
