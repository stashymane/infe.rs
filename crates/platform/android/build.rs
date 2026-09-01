fn main() {
    // AHardwareBuffer_* and AHardwareBuffer_fromHardwareBuffer live in libandroid.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("android") {
        println!("cargo:rustc-link-lib=dylib=android");
    }
}
