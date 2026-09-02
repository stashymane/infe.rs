use std::env;
use std::path::{Path, PathBuf};

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

/// Core ExecuTorch static libraries required by `executorch-sys` (mirrors its build.rs).
fn link_executorch(libs_dir: &Path) {
    require_file(
        &libs_dir.join("libexecutorch.a"),
        "executorch",
        "libexecutorch.a",
    );

    println!("cargo:rustc-link-search=native={}", libs_dir.display());
    println!("cargo:rustc-link-lib=static:+whole-archive=executorch");
    println!("cargo:rustc-link-lib=static:+whole-archive=executorch_core");

    let data_loader = libs_dir.join("extension/data_loader");
    require_file(
        &data_loader.join("libextension_data_loader.a"),
        "executorch",
        "libextension_data_loader.a",
    );
    println!("cargo:rustc-link-search=native={}", data_loader.display());
    println!("cargo:rustc-link-lib=static:+whole-archive=extension_data_loader");

    let module = libs_dir.join("extension/module");
    require_file(
        &module.join("libextension_module_static.a"),
        "executorch",
        "libextension_module_static.a",
    );
    println!("cargo:rustc-link-search=native={}", module.display());
    println!("cargo:rustc-link-lib=static:+whole-archive=extension_module_static");

    let named_data_map = libs_dir.join("extension/named_data_map");
    require_file(
        &named_data_map.join("libextension_named_data_map.a"),
        "executorch",
        "libextension_named_data_map.a",
    );
    println!("cargo:rustc-link-search=native={}", named_data_map.display());
    println!("cargo:rustc-link-lib=static:+whole-archive=extension_named_data_map");

    let flat_tensor = libs_dir.join("extension/flat_tensor");
    require_file(
        &flat_tensor.join("libextension_flat_tensor.a"),
        "executorch",
        "libextension_flat_tensor.a",
    );
    println!("cargo:rustc-link-search=native={}", flat_tensor.display());
    println!("cargo:rustc-link-lib=static:+whole-archive=extension_flat_tensor");

    let tensor = libs_dir.join("extension/tensor");
    require_file(
        &tensor.join("libextension_tensor.a"),
        "executorch",
        "libextension_tensor.a",
    );
    println!("cargo:rustc-link-search=native={}", tensor.display());
    println!("cargo:rustc-link-lib=static:+whole-archive=extension_tensor");
}

fn link_portable(libs_dir: &Path) {
    let portable = libs_dir.join("kernels/portable");
    let ops = portable.join("libportable_ops_lib.a");
    let kernels = portable.join("libportable_kernels.a");
    require_file(&ops, "portable", "libportable_ops_lib.a");
    require_file(&kernels, "portable", "libportable_kernels.a");

    println!("cargo:rustc-link-search=native={}", portable.display());
    println!("cargo:rustc-link-lib=static:+whole-archive=portable_ops_lib");
    println!("cargo:rustc-link-lib=static:+whole-archive=portable_kernels");
}

fn link_xnnpack(libs_dir: &Path) {
    let xnnpack_dir = libs_dir.join("backends/xnnpack");
    let backend = xnnpack_dir.join("libxnnpack_backend.a");
    require_file(&backend, "xnnpack", "libxnnpack_backend.a");

    let threadpool = libs_dir.join("extension/threadpool/libextension_threadpool.a");
    require_file(&threadpool, "xnnpack", "libextension_threadpool.a");

    let deps = [
        (
            xnnpack_dir.join("third-party/pthreadpool/libpthreadpool.a"),
            "libpthreadpool.a",
        ),
        (
            xnnpack_dir.join("third-party/cpuinfo/libcpuinfo.a"),
            "libcpuinfo.a",
        ),
        (
            xnnpack_dir.join("third-party/XNNPACK/libXNNPACK.a"),
            "libXNNPACK.a",
        ),
        (
            xnnpack_dir.join("third-party/XNNPACK/libxnnpack-microkernels-prod.a"),
            "libxnnpack-microkernels-prod.a",
        ),
    ];
    for (path, name) in deps {
        require_file(&path, "xnnpack", name);
    }

    println!(
        "cargo:rustc-link-search=native={}",
        libs_dir.join("extension/threadpool").display()
    );
    println!("cargo:rustc-link-lib=static=extension_threadpool");

    println!("cargo:rustc-link-search=native={}", xnnpack_dir.display());
    println!("cargo:rustc-link-lib=static:+whole-archive=xnnpack_backend");

    for sub in [
        "third-party/pthreadpool",
        "third-party/cpuinfo",
        "third-party/XNNPACK",
    ] {
        let p = xnnpack_dir.join(sub);
        println!("cargo:rustc-link-search=native={}", p.display());
    }
    for lib in ["pthreadpool", "cpuinfo", "XNNPACK", "xnnpack-microkernels-prod"] {
        println!("cargo:rustc-link-lib=static={lib}");
    }

    // Android/arm64 XNNPACK builds pull KleidiAI microkernels; link when present.
    let kleidiai = libs_dir.join("kleidiai/libkleidiai.a");
    if kleidiai.exists() {
        println!(
            "cargo:rustc-link-search=native={}",
            libs_dir.join("kleidiai").display()
        );
        println!("cargo:rustc-link-lib=static=kleidiai");
    }
}

fn link_vulkan(libs_dir: &Path) {
    let vulkan_dir = libs_dir.join("backends/vulkan");
    let vulkan_backend = vulkan_dir.join("libvulkan_backend.a");
    let vulkan_ffi = vulkan_dir.join("libinfers_et_vulkan_ffi.a");
    require_file(&vulkan_backend, "vulkan", "libvulkan_backend.a");
    require_file(
        &vulkan_ffi,
        "vulkan",
        "libinfers_et_vulkan_ffi.a (Nix-built Vulkan FFI)",
    );

    println!("cargo:rustc-link-search=native={}", vulkan_dir.display());
    println!("cargo:rustc-link-lib=static=infers_et_vulkan_ffi");
    println!("cargo:rustc-link-lib=static:+whole-archive=vulkan_backend");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_vendor = env::var("CARGO_CFG_TARGET_VENDOR").unwrap_or_default();
    if target_os == "linux" && target_vendor != "android" {
        println!("cargo:rustc-link-lib=dylib=vulkan");
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}
