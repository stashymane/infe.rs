use spirv_builder::{Capability, SpirvBuilder, SpirvMetadata};
use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let result = SpirvBuilder::new("shaders", "spirv-unknown-vulkan1.1")
        .spirv_metadata(SpirvMetadata::Full)
        .capability(Capability::Int8)
        .build()
        .unwrap();

    let spv_path = match result.module {
        spirv_builder::ModuleResult::SingleModule(path) => path,
        spirv_builder::ModuleResult::MultiModule(_) => panic!("expected single module"),
    };

    let out_dir = env::var_os("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("processing_shaders.spv");
    fs::copy(&spv_path, &dest_path).unwrap();

    println!("cargo:rustc-env=PROCESSING_SHADERS_SPV={}", dest_path.display());
}
