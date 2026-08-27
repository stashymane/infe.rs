# infe.rs

ML pipeline assembly kit. Aims to provide the most optimal, performant way to run any combination of models, modifiable
at runtime, with as little copying as needed.

Built on ExecuTorch with custom patches. Mostly aimed at Android for now.

## Usage

Currently only available via git, not yet ready for production use.

```toml
[dependencies]
infers = { git = "...", features = ["..."] }
```

### Features

| Feature    | Default | Effect                                                                  |
|------------|---------|-------------------------------------------------------------------------|
| `portable` | ✓      | ExecuTorch portable CPU delegate                                        |
| `xnnpack`  | ✓      | XNNPACK CPU acceleration                                                |
| `vulkan`   | ✓      | Vulkan ExecuTorch delegate, `GpuImageProcessor`, shared `VulkanContext` |

## Benchmarks/examples

Benchmarks are available in `examples/` which time how long preprocessing and inference take on the CPU or Vulkan.

```bash
# CPU: preprocess -> NHWC→NCHW -> XNNPACK yolo26n-face
cargo run --release --example cpu_pipeline_bench
cargo run --release --example cpu_pipeline_bench -- --iters 100 --threads 4

# GPU: Vulkan preprocess -> NHWC->NCHW -> Vulkan ExecuTorch
cargo run --release --example gpu_pipeline_bench
cargo run --release --example gpu_pipeline_bench -- --iters 100
```

By default these load `yolo26n-face`, which is not bundled in this repo - you can build the models with a script in the
`scripts/` directory.

## Development

### Requirements

* Android NDK
* Rust toolchain (specified in `rust-toolchain.toml`)
* Docker (for building dependencies)

### Steps

Before this can compile, you have to build ExecuTorch (+ yolo26n-face for testing). This is automated by
`scripts/build_assets.sh` - builds run within a Docker container, only outputting the necessary assets.

Once the assets are built, everything else is standard Cargo.

For information about the Kotlin Multiplatform module, check out `kotlin/README.md`.
