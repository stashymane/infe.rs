use std::env;
use std::path::{Path, PathBuf};

include!("../../../build/android_ndk.rs");
include!("../../../build/executorch_lib_dir.rs");

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf();

    let env_root = env::var("EXECUTORCH_RS_EXECUTORCH_LIB_DIR")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_root.join("target/executorch"));

    let libs_dir = resolve_executorch_lib_dir(env_root, &workspace_root);

    println!("cargo:rerun-if-env-changed=EXECUTORCH_RS_EXECUTORCH_LIB_DIR");

    link_executorch(&libs_dir);

    if feature_enabled("portable") {
        link_portable(&libs_dir);
    }

    if feature_enabled("xnnpack") {
        link_xnnpack(&libs_dir);
    }

    if feature_enabled("vulkan") {
        link_vulkan(&libs_dir);
    }

    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("android") {
        link_executorch_android();
    }
}

fn feature_enabled(name: &str) -> bool {
    let key = format!("CARGO_FEATURE_{}", name.to_uppercase().replace('-', "_"));
    env::var(key).is_ok()
}

fn require_file(path: &Path, feature: &str, description: &str) {
    if path.exists() {
        return;
    }
    panic!(
        "ExecuTorch `{feature}` feature is enabled but {description} was not found at {}.\n\
         Build ExecuTorch libraries (see scripts/build_executorch.sh) or disable \
         the `{feature}` feature on `infers-backend-executorch`.",
        path.display()
    );
}

fn link_static(dir: &Path, lib: &str, whole_archive: bool) {
    println!("cargo:rustc-link-search=native={}", dir.display());
    if whole_archive {
        println!("cargo:rustc-link-lib=static:+whole-archive={lib}");
    } else {
        println!("cargo:rustc-link-lib=static={lib}");
    }
}

fn link_archive(dir: &Path, feature: &str, file: &str, lib: &str, whole_archive: bool) {
    require_file(&dir.join(file), feature, file);
    link_static(dir, lib, whole_archive);
}

fn link_executorch(libs_dir: &Path) {
    link_archive(libs_dir, "executorch", "libexecutorch.a", "executorch", true);
    link_static(libs_dir, "executorch_core", true);

    for (subdir, file, lib) in [
        (
            "extension/data_loader",
            "libextension_data_loader.a",
            "extension_data_loader",
        ),
        (
            "extension/module",
            "libextension_module_static.a",
            "extension_module_static",
        ),
        (
            "extension/named_data_map",
            "libextension_named_data_map.a",
            "extension_named_data_map",
        ),
        (
            "extension/flat_tensor",
            "libextension_flat_tensor.a",
            "extension_flat_tensor",
        ),
        (
            "extension/tensor",
            "libextension_tensor.a",
            "extension_tensor",
        ),
    ] {
        link_archive(&libs_dir.join(subdir), "executorch", file, lib, true);
    }
}

fn link_portable(libs_dir: &Path) {
    let portable = libs_dir.join("kernels/portable");
    link_archive(
        &portable,
        "portable",
        "libportable_ops_lib.a",
        "portable_ops_lib",
        true,
    );
    link_archive(
        &portable,
        "portable",
        "libportable_kernels.a",
        "portable_kernels",
        true,
    );
}

fn link_xnnpack(libs_dir: &Path) {
    let xnnpack_dir = libs_dir.join("backends/xnnpack");
    link_archive(
        &xnnpack_dir,
        "xnnpack",
        "libxnnpack_backend.a",
        "xnnpack_backend",
        true,
    );
    link_archive(
        &libs_dir.join("extension/threadpool"),
        "xnnpack",
        "libextension_threadpool.a",
        "extension_threadpool",
        false,
    );

    for (subdir, file, lib) in [
        (
            "third-party/pthreadpool",
            "libpthreadpool.a",
            "pthreadpool",
        ),
        ("third-party/cpuinfo", "libcpuinfo.a", "cpuinfo"),
        ("third-party/XNNPACK", "libXNNPACK.a", "XNNPACK"),
        (
            "third-party/XNNPACK",
            "libxnnpack-microkernels-prod.a",
            "xnnpack-microkernels-prod",
        ),
    ] {
        link_archive(&xnnpack_dir.join(subdir), "xnnpack", file, lib, false);
    }

    let kleidiai = libs_dir.join("kleidiai/libkleidiai.a");
    if kleidiai.exists() {
        link_archive(&libs_dir.join("kleidiai"), "xnnpack", "libkleidiai.a", "kleidiai", false);
    }
}

fn link_vulkan(libs_dir: &Path) {
    let vulkan_dir = libs_dir.join("backends/vulkan");
    link_archive(
        &vulkan_dir,
        "vulkan",
        "libinfers_et_vulkan_ffi.a",
        "infers_et_vulkan_ffi",
        false,
    );
    link_archive(
        &vulkan_dir,
        "vulkan",
        "libvulkan_backend.a",
        "vulkan_backend",
        true,
    );

    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux")
        && env::var("CARGO_CFG_TARGET_VENDOR").as_deref() != Ok("android")
    {
        println!("cargo:rustc-link-lib=dylib=vulkan");
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}
