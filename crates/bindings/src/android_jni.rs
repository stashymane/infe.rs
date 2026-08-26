//! JNI bridge so Kotlin can obtain an `AHardwareBuffer*` from
//! `android.hardware.HardwareBuffer` (which exposes no public native handle).

use std::ffi::c_void;

/// Returns the raw `AHardwareBuffer*` as a `jlong`, or `0` on failure.
///
/// Prefer letting `create_hardware_buffer_from_raw` (`from_raw` / acquire) own a
/// separate +1 for the Rust handle. Do not pair this with `nativeRelease` when
/// the Java `HardwareBuffer` remains responsible for closing.
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

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_dev_stashy_infers_HardwareBufferBridge_nativeRelease(
    _env: *mut c_void,
    _class: *mut c_void,
    ptr: i64,
) {
    if ptr == 0 {
        return;
    }
    // SAFETY: `ptr` is an `AHardwareBuffer*` previously returned by
    // `AHardwareBuffer_fromHardwareBuffer` with a live +1 ref.
    unsafe {
        platform_android::ffi::AHardwareBuffer_release(ptr as *mut platform_android::ffi::AHardwareBuffer);
    }
}
