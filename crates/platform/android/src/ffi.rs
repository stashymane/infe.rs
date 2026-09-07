#[allow(unused_imports)]
use std::ffi::c_void;

// Opaque AHardwareBuffer struct
#[repr(C)]
pub struct AHardwareBuffer {
    _unused: [u8; 0],
}

// Android HardwareBuffer Format constants (matching android/hardware_buffer.h)
pub const AHARDWAREBUFFER_FORMAT_R8G8B8_UNORM: u32 = 3;
#[allow(non_upper_case_globals)]
pub const AHARDWAREBUFFER_FORMAT_Y8Cb8Cr8_420: u32 = 0x23;

// Android HardwareBuffer Usage Flags
pub const AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN: u64 = 3;
pub const AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN: u64 = 3 << 4;
pub const AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE: u64 = 1 << 8;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ARect {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct AHardwareBuffer_Desc {
    pub width: u32,
    pub height: u32,
    pub layers: u32,
    pub format: u32,
    pub usage: u64,
    pub stride: u32,
    pub rfu0: u32,
    pub rfu1: u64,
}

// Android NDK functions declarations
#[cfg(target_os = "android")]
unsafe extern "C" {
    pub fn AHardwareBuffer_allocate(
        desc: *const AHardwareBuffer_Desc,
        outBuffer: *mut *mut AHardwareBuffer,
    ) -> i32;

    pub fn AHardwareBuffer_acquire(buffer: *mut AHardwareBuffer);

    pub fn AHardwareBuffer_release(buffer: *mut AHardwareBuffer);

    pub fn AHardwareBuffer_describe(
        buffer: *const AHardwareBuffer,
        outDesc: *mut AHardwareBuffer_Desc,
    );

    pub fn AHardwareBuffer_lock(
        buffer: *mut AHardwareBuffer,
        usage: u64,
        fence: i32,
        rect: *const ARect,
        outVirtualAddress: *mut *mut c_void,
    ) -> i32;

    pub fn AHardwareBuffer_unlock(buffer: *mut AHardwareBuffer, fence: *mut i32) -> i32;

    /// Return the `AHardwareBuffer` wrapped by a Java `HardwareBuffer`.
    ///
    /// Per NDK docs this does **not** acquire an additional reference. The
    /// pointer stays valid only while the Java object keeps the buffer alive;
    /// call `AHardwareBuffer_acquire` before the Java object is closed if the
    /// native side needs independent ownership.
    pub fn AHardwareBuffer_fromHardwareBuffer(
        env: *mut c_void,
        hardwareBuffer: *mut c_void,
    ) -> *mut AHardwareBuffer;
}
