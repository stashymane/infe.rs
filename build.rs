use std::path::Path;

fn main() {
    // Benches `include_bytes!` these; rebuild when the exported programs change.
    for rel in [
        "target/yolo26n-face/xnnpack/model.pte",
        "target/yolo26n-face/vulkan/model.pte",
    ] {
        println!("cargo:rerun-if-changed={rel}");
        if !Path::new(rel).is_file() {
            println!(
                "cargo:warning=missing {rel}; build with scripts/build_assets.sh before tests/benchmarks"
            );
        }
    }
}
