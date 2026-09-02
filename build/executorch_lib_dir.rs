pub fn resolve_executorch_lib_dir(
    root: PathBuf,
    workspace_root: &Path,
) -> PathBuf {
    let root = if root.as_os_str().is_empty() {
        workspace_root.join("target/executorch")
    } else {
        root
    };

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    let subdir = match (target_os.as_str(), target_arch.as_str()) {
        ("android", "aarch64") => "android-arm64",
        ("linux", "x86_64") => "x86_64-unknown-linux-gnu",
        (os, arch) => panic!(
            "No ExecuTorch prebuilt tree for target {arch}-{os} under {}.\n\
             Build assets with scripts/build_executorch.sh. \
             EXECUTORCH_RS_EXECUTORCH_LIB_DIR must point at the ExecuTorch root \
             (target/executorch), not a per-target subdirectory.",
            root.display()
        ),
    };

    root.join(subdir)
}
