use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf();

    let libs_dir = env::var("EXECUTORCH_RS_EXECUTORCH_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| workspace_root.join("target/executorch-libs"));

    println!("cargo:rerun-if-env-changed=EXECUTORCH_RS_EXECUTORCH_LIB_DIR");
    println!("cargo:rerun-if-env-changed=EXECUTORCH_SRC");
    println!("cargo:rerun-if-changed=cpp/vulkan_external_adapter.cpp");
    println!("cargo:rerun-if-changed=cpp/vulkan_gpu_input.cpp");

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
}

fn executorch_src_dir() -> Option<PathBuf> {
    let src = env::var("EXECUTORCH_SRC").ok()?;
    let path = PathBuf::from(src);
    if path
        .join("src/executorch/backends/vulkan/runtime/graph/ComputeGraph.h")
        .exists()
    {
        Some(path)
    } else {
        None
    }
}

fn link_vulkan(libs_dir: &Path) {
    let vulkan_backend = libs_dir.join("backends/vulkan/libvulkan_backend.a");
    require_file(&vulkan_backend, "vulkan", "libvulkan_backend.a");

    let mut builder = cc::Build::new();
    builder
        .cpp(true)
        .std("c++17")
        .file("cpp/vulkan_external_adapter.cpp")
        .file("cpp/vulkan_gpu_input.cpp")
        .flag_if_supported("-Wno-unused-parameter");

    if let Some(src) = executorch_src_dir() {
        println!("cargo:rerun-if-changed={}", src.display());
        builder.define("INFERS_ET_EXECUTORCH_SRC", None);
        builder.include(src.join("src"));
        builder.include(src.join("third-party/Vulkan-Headers/include"));
        builder.include(src.join("third-party/volk"));
        builder.include(src.join("third-party/VulkanMemoryAllocator"));
    }

    builder.compile("infers_et_vulkan_ffi");

    println!(
        "cargo:rustc-link-search=native={}",
        libs_dir.join("backends/vulkan").display()
    );
    println!("cargo:rustc-link-lib=static:+whole-archive=vulkan_backend");
    println!("cargo:rustc-link-lib=dylib=vulkan");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
