# infe.rs

ML pipeline assembly kit. Aims to provide the most optimal, performant way to run any combination of models, modifiable
at runtime, with as little copying as needed.

Built on ExecuTorch with custom patches. Mostly aimed at Android for now.

## Usage

> [!WARNING]
> This library is experimental, no stable API/ABI is provided yet.

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

## Benchmarks

By default, these load `yolo26n-face`, which is not bundled in this repo — build the models with
`scripts/build_yolo26n_face.sh` (or just `scripts/build_assets.sh`).

```bash
# CPU
cargo bench --bench cpu_pipeline

# GPU
cargo bench --bench gpu_pipeline

# Run a single stage
cargo bench --bench cpu_pipeline -- preprocess
cargo bench --bench cpu_pipeline -- inference
cargo bench --bench cpu_pipeline -- full_pass
```

## Development

### Requirements

* Nix (with flake support)
* Rust toolchain (specified in `rust-toolchain.toml`)
    * `cargo-ndk` if working with Android
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
