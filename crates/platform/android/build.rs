fn main() {
    // AHardwareBuffer_* and AHardwareBuffer_fromHardwareBuffer live in libandroid.
    println!("cargo:rustc-link-lib=dylib=android");
}
