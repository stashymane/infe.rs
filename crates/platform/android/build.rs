use std::env;
use std::path::PathBuf;

include!("../../../build/android_ndk.rs");

fn main() {
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("android") {
        link_hardware_buffer();
    }
}
