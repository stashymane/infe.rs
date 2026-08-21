#[allow(unused_imports)]
use std::ffi::c_void;

// Opaque AHardwareBuffer struct
#[repr(C)]
pub struct AHardwareBuffer {
    _unused: [u8; 0],
}

// Android HardwareBuffer Format constants (matching android/hardware_buffer.h)
pub const AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM: u32 = 1;
pub const AHARDWAREBUFFER_FORMAT_R8G8B8X8_UNORM: u32 = 2;
pub const AHARDWAREBUFFER_FORMAT_R8G8B8_UNORM: u32 = 3;
pub const AHARDWAREBUFFER_FORMAT_R5G6B5_UNORM: u32 = 4;
pub const AHARDWAREBUFFER_FORMAT_R16G16B16A16_FLOAT: u32 = 0x16;
pub const AHARDWAREBUFFER_FORMAT_RGBA_FP16: u32 = 0x16;
pub const AHARDWAREBUFFER_FORMAT_RGBA_1010102: u32 = 0x2b;
pub const AHARDWAREBUFFER_FORMAT_BLOB: u32 = 0x21;
#[allow(non_upper_case_globals)]
pub const AHARDWAREBUFFER_FORMAT_Y8Cb8Cr8_420: u32 = 0x23;

// Android HardwareBuffer Usage Flags
pub const AHARDWAREBUFFER_USAGE_CPU_READ_NEVER: u64 = 0;
pub const AHARDWAREBUFFER_USAGE_CPU_READ_RARELY: u64 = 2;
pub const AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN: u64 = 3;
pub const AHARDWAREBUFFER_USAGE_CPU_READ_MASK: u64 = 0xF;
pub const AHARDWAREBUFFER_USAGE_CPU_WRITE_NEVER: u64 = 0;
pub const AHARDWAREBUFFER_USAGE_CPU_WRITE_RARELY: u64 = 2 << 4;
pub const AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN: u64 = 3 << 4;
pub const AHARDWAREBUFFER_USAGE_CPU_WRITE_MASK: u64 = 0xF << 4;
pub const AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE: u64 = 1 << 8;
pub const AHARDWAREBUFFER_USAGE_GPU_COLOR_OUTPUT: u64 = 1 << 9;
pub const AHARDWAREBUFFER_USAGE_PROTECTED_CONTENT: u64 = 1 << 14;
pub const AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER: u64 = 1 << 24;

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
extern "C" {
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
}
