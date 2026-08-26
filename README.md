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

## Kotlin Multiplatform (`kotlin/`)

Android + JVM desktop hosts consume `infers-bindings` through Gobley (UniFFI). There is **one** native library (`libinfers_bindings`); Cargo features are selected at build time, not via multiple `.so` files.

Public modules hand-write wrappers over the generated `dev.stashy.infers.ffi` surface. `:infers-ffi` is an `implementation` dependency, so apps never see generated types on their compile classpath.

### Suspending, frame-scoped API

Blocking `process` / `run` / `loadModel(ByteArray)` are not part of the public surface. Long-lived handles (`Backend`, `ModelSession`, image processors) stay `AutoCloseable`. Per-frame tensors are created inside `inferenceScope` and closed automatically when the scope exits:

```kotlin
Backend().use { backend ->
    backend.loadModel(Path("model.pte"), XnnpackConfig()).use { session ->
        CpuImageProcessor().use { processor ->
            inferenceScope {
                val input = processor.process(frameBytes, options)
                val outputs = session.run(input)
                render(outputs.first().readFloats())
            }
        }
    }
}
```

`loadModel` takes a kotlinx-io `Path`; the file must stay readable for the session lifetime. Callers that only have a stream should write their own file. Camera pipelines typically use `conflate().mapInference { … }` so stale frames are dropped; native calls are not interruptible mid-inference.

JVM desktop embeds the host native library only (for local tests). Android AAR multi-ABI publishing is unchanged; multi-OS desktop classifier JARs are a later follow-up.

### Modules

| Module | Role |
|--------|------|
| `:infers` | Public KMP core (`Device`, `Tensor`, `CpuImageProcessor`, `Backend`, …). Android extensions (`HardwareBuffer`) live in `androidMain`. |
| `:infers-ffi` | Builds `crates/bindings` + generates UniFFI Kotlin (internal) |
| `:infers-portable` / `:infers-xnnpack` / `:infers-vulkan` | One module per Cargo feature; each expands the public API for that feature |

### Consumer setup

1. Include the Gradle project (composite build or `include` from an app).
2. Set `infers.cargo.features` in `kotlin/gradle.properties` (or `-P`) to the Cargo features you want, e.g. `portable,xnnpack` or `portable,xnnpack,vulkan`.
3. Depend on the matching modules:

```kotlin
dependencies {
    implementation(project(":infers"))
    implementation(project(":infers-xnnpack"))
    implementation(project(":infers-portable"))
    // optional:
    implementation(project(":infers-vulkan"))
}
```

Shared APIs belong in `commonMain`. Android-only APIs (e.g. `HardwareBuffer`) are only visible from `androidMain` of the module that ships them.

### Requirements

- JDK compatible with the Gradle wrapper (Gradle 9.5+; Kotlin 2.4.x supports up to 9.5 fully).
- Android SDK + NDK for Android targets (`kotlin/local.properties` → `sdk.dir=...`).
- Rust toolchain able to build `infers-bindings` (same as the Cargo workspace).
- Android `.so` builds need ExecuTorch static libs built for the Android ABI (host libs under `target/executorch-libs` are not sufficient for `cargoBuildAndroid*`). Build them with Docker only: `./scripts/build_executorch.sh android-arm64` → `target/executorch-libs-android-arm64`.
- Gobley **0.3.7** requires AGP **8.x** (`com.android.library`); AGP 9 / `com.android.kotlin.multiplatform.library` needs Gobley 0.4+.

Android/Kotlin Gradle builds run via Docker (SDK, NDK, Rust, and JDK are in the image):

```bash
./scripts/android_gradle.sh :infers-ffi:printInfersCargoFeatures
./scripts/android_gradle.sh :infers:jvmTest :infers-xnnpack:jvmTest :infers-vulkan:jvmTest
./scripts/android_gradle.sh :infers-consumer-test:compileKotlinJvm
./scripts/android_gradle.sh publishToMavenLocal
# HardwareBuffer instrumented tests; needs a device/emulator visible to host adb:
./scripts/android_gradle.sh :infers:connectedDebugAndroidTest :infers-vulkan:connectedDebugAndroidTest
```
