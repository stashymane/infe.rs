use spirv_builder::{Capability, SpirvBuilder, SpirvMetadata};
use std::env;
use std::fs;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=../shaders");
    println!("cargo:rerun-if-changed=../shaders/src/lib.rs");
    println!("cargo:rerun-if-changed=../core/src/options.rs");

    let out_dir = env::var_os("OUT_DIR").unwrap();
    let spirv_target = Path::new(&out_dir).join("spirv-builder");
    let result = SpirvBuilder::new("../shaders", "spirv-unknown-vulkan1.1")
        .spirv_metadata(SpirvMetadata::Full)
        .capability(Capability::Int8)
        .target_dir_path(spirv_target)
        .build()
        .unwrap();

    let spv_path = match result.module {
        spirv_builder::ModuleResult::SingleModule(path) => path,
        spirv_builder::ModuleResult::MultiModule(_) => panic!("expected single module"),
    };

    let dest_path = Path::new(&out_dir).join("processing_shaders.spv");
    fs::copy(&spv_path, &dest_path).unwrap();

    println!("cargo:rustc-env=PROCESSING_SHADERS_SPV={}", dest_path.display());
}
