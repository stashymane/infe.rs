//! JNI bridge so Kotlin can obtain an `AHardwareBuffer*` from
//! `android.hardware.HardwareBuffer` (which exposes no public native handle).

use std::ffi::c_void;

/// Returns the raw `AHardwareBuffer*` as a `jlong`, or `0` on failure.
///
/// Per NDK, `AHardwareBuffer_fromHardwareBuffer` does **not** acquire. The
/// pointer is borrowed from the Java `HardwareBuffer` and stays valid only while
/// that object keeps the buffer alive. Pass it to
/// `create_hardware_buffer_from_java`, which acquires an independent +1 for Rust.
/// Never `AHardwareBuffer_release` this borrowed pointer — that would drop the
/// Java object's reference.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_dev_stashy_infers_HardwareBufferBridge_nativePointer(
    env: *mut c_void,
    _class: *mut c_void,
    buffer: *mut c_void,
) -> i64 {
    if env.is_null() || buffer.is_null() {
        return 0;
    }
    // SAFETY: `env`/`buffer` are the JNIEnv* and jobject from the Android runtime.
    let ptr = unsafe { platform_android::ffi::AHardwareBuffer_fromHardwareBuffer(env, buffer) };
    ptr as i64
}
