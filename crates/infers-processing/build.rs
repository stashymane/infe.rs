fn main() {
    #[cfg(feature = "vulkan")]
    build_shaders();
}

#[cfg(feature = "vulkan")]
fn build_shaders() {
    use spirv_builder::{Capability, SpirvBuilder, SpirvMetadata};
    use std::env;
    use std::fs;
    use std::path::Path;

    println!("cargo:rerun-if-changed=../infers-processing-shaders");
    println!("cargo:rerun-if-changed=../infers-processing-shaders/src/lib.rs");
    println!("cargo:rerun-if-changed=../infers-processing-core/src/options.rs");
    println!("cargo:rerun-if-changed=../infers-processing-core/src/convert.rs");
    println!("cargo:rerun-if-changed=../infers-processing-core/src/lib.rs");

    let out_dir = env::var_os("OUT_DIR").unwrap();
    let spirv_target = Path::new(&out_dir).join("spirv-builder");
    let result = SpirvBuilder::new("../infers-processing-shaders", "spirv-unknown-vulkan1.1")
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
