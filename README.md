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
# CPU: preprocess (NCHW) -> XNNPACK yolo26n-face
cargo run --release --example cpu_pipeline_bench
cargo run --release --example cpu_pipeline_bench -- --iters 100 --threads 4

# GPU: Vulkan preprocess (NCHW on GPU) -> Vulkan ExecuTorch
cargo run --release --example gpu_pipeline_bench
cargo run --release --example gpu_pipeline_bench -- --iters 100
```

By default, these load `yolo26n-face`, which is not bundled in this repo — build the models with
`scripts/build_yolo26n_face.sh`.

## Development

### Requirements

* Nix (with flake support)
* Rust toolchain (specified in `rust-toolchain.toml`)
* JDK + Gradle (for Kotlin library)

### Steps

Before this can compile, you have to build ExecuTorch (+ yolo26n-face for testing).
`scripts/build_assets.sh` does this all for you.

You may also build these assets directly:

```bash
nix build .#executorch-x86_64-unknown-linux-gnu --out-link target/executorch/x86_64-unknown-linux-gnu
nix build .#executorch-android-arm64 --out-link target/executorch/android-arm64
# or both targets:
nix build .#executorch-libs --out-link target/executorch
```

yolo26n-face model (configure via `assets/yolo26n-face-manifest.yaml`):

```bash
nix build .#yolo26n-face-export --out-link target/yolo26n-face
```

Once the assets are built, everything else is standard Cargo.

For information about the Kotlin Multiplatform module, check out `kotlin/README.md`.
