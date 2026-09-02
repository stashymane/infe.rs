const MIN_API: u32 = 26;

fn ndk_lib_dir(api: Option<u32>) -> Option<PathBuf> {
    let (sysroot, triple) = if let (Ok(sysroot), Ok(triple)) = (
        env::var("CARGO_NDK_SYSROOT_PATH"),
        env::var("CARGO_NDK_SYSROOT_TARGET"),
    ) {
        (sysroot, triple)
    } else {
        let home = env::var("ANDROID_NDK_HOME")
            .or_else(|_| env::var("ANDROID_NDK_ROOT"))
            .ok()?;
        let host = ndk_host_tag()?;
        let arch = env::var("CARGO_CFG_TARGET_ARCH").ok()?;
        (
            format!(
                "{}/toolchains/llvm/prebuilt/{}/sysroot",
                home, host
            ),
            format!("{arch}-linux-android"),
        )
    };

    let mut dir = PathBuf::from(sysroot)
        .join("usr/lib")
        .join(triple);
    if let Some(api) = api {
        dir.push(api.to_string());
    }
    Some(dir)
}

pub fn link_hardware_buffer() {
    if let Some(dir) = ndk_lib_dir(Some(MIN_API)) {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    println!("cargo:rustc-link-lib=dylib=android");
}

pub fn link_executorch_android() {
    if let Some(dir) = ndk_lib_dir(Some(MIN_API)) {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    if let Some(dir) = ndk_lib_dir(None) {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    println!("cargo:rustc-link-lib=static=c++_static");
    println!("cargo:rustc-link-lib=static=c++abi");
    println!("cargo:rustc-link-lib=dylib=log");
}

fn ndk_host_tag() -> Option<&'static str> {
    match env::var("HOST").ok()?.as_str() {
        host if host.ends_with("-linux-gnu") => Some("linux-x86_64"),
        "aarch64-apple-darwin" => Some("darwin-arm64"),
        "x86_64-apple-darwin" => Some("darwin-x86_64"),
        _ => None,
    }
}
