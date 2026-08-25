# Infers

On-device machine learning inference for Rust. Infers wraps [ExecuTorch](https://pytorch.org/executorch/) for model execution and provides Vulkan-accelerated image preprocessing so camera frames can stay on the GPU from resize/normalize through inference.

## Workspace crates

| Crate | Purpose | When to depend on it |
|-------|---------|----------------------|
| `infers` | Meta-crate: re-exports core, processing, and ExecuTorch backend | Default choice for Rust apps |
| `infers-core` | Tensors, devices, image buffers, backend traits | Custom backends or minimal dependencies |
| `processing` | CPU and GPU image processors | Preprocessing only |
| `infers-backend-executorch` | ExecuTorch `Backend` implementation | Direct ExecuTorch integration |
| `infers-bindings` | UniFFI bindings (`cdylib`) for Kotlin/Swift | Mobile or foreign-language hosts |
| `infers-gpu` | Shared Vulkan context and buffers | Low-level GPU utilities |
| `platform-android` | `AHardwareBuffer` integration | Android camera pipelines |
| `infers-test-utils` | Mock backend/session helpers | Tests and examples |

## Feature flags

The root `infers` crate and `infers-bindings` share the same defaults:

| Feature | Effect |
|---------|--------|
| `portable` (default) | ExecuTorch portable CPU delegate |
| `xnnpack` (default) | XNNPACK CPU acceleration |
| `vulkan` (default) | Vulkan ExecuTorch delegate, `GpuImageProcessor`, shared `VulkanContext` |

Disable defaults and enable only what you need, for example:

```toml
[dependencies]
infers = { path = "...", default-features = false, features = ["xnnpack"] }
```

## Pipeline sketch

```
Camera / image buffer (Rgb888, Nv12, …)
        ↓
  CpuImageProcessor or GpuImageProcessor
  (crop, fit: Contain / Stretch / Crop, rotation: Rot90, …)
        ↓
  Model tensor on CPU or GPU ([1, H, W, 3] Rgbf32)
        ↓
  ExecuTorchSession::run (Xnnpack, Vulkan, …)
        ↓
  Output tensors → explicit read_to_cpu when needed
```

On Android, `AndroidHardwareBufferHandle` can feed the GPU processor directly. With `vulkan`, the same `VulkanContext` should be shared between `GpuImageProcessor` and `ExecuTorchBackendConfig::Vulkan` so preprocessing and inference use one device.

## Benchmark examples

Release-mode examples time the hot path after a camera-like frame is already resident (allocation is outside the timer):

```bash
# CPU: preprocess → NHWC→NCHW → XNNPACK yolo26n-face
cargo run --release --example cpu_pipeline_bench
cargo run --release --example cpu_pipeline_bench -- --iters 100 --threads 4

# GPU: Vulkan preprocess → NHWC→NCHW → Vulkan ExecuTorch
cargo run --release --example gpu_pipeline_bench
cargo run --release --example gpu_pipeline_bench -- --iters 100
```

Defaults load `assets/yolo26n-face/{xnnpack,vulkan}/model.pte` (192²). Flags are parsed with `clap` (`--model`, `--width` / `--height`, `--warmup`, `--iters`, `--threads` on CPU, `--help`). Reported stages: preprocess, layout convert, inference, and total.

## Development

Set a stable target directory (optional but recommended in this repo):

```bash
export CARGO_TARGET_DIR=/mnt/ssd/Projects/Rust/infers/target
```

Run the full test suite:

```bash
cargo test --workspace
```

Run Clippy with warnings denied:

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

Integration tests in `tests/` and `crates/bindings/tests/` exercise multi-stage face-detection style pipelines with mock backends. GPU and Vulkan tests skip automatically when no device is available.

## Errors

Public APIs surface typed errors: `CoreError` (core), `GpuError` (Vulkan), and `InfersError` (UniFFI bindings). Image and tensor enums use Rust-style names (`Rgb888`, `Rgbf32`, `Nv12`, `Contain`, `Rot90`, etc.).
