#!/usr/bin/env bash
set -euo pipefail

EXECUTORCH_VERSION="${EXECUTORCH_VERSION:-v1.4.0}"
EXECUTORCH_DIR="${EXECUTORCH_DIR:-/workspace/executorch}"
BUILD_DIR="${BUILD_DIR:-/workspace/build}"
OUTPUT_DIR="${OUTPUT_DIR:-/output}"
# host (default) | android-arm64
TARGET="${TARGET:-host}"
ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-/opt/android-ndk}"
ANDROID_NDK_API="${ANDROID_NDK_API:-26}"
ANDROID_ABI="${ANDROID_ABI:-arm64-v8a}"
# Set FORCE_CLONE=1 to discard a persisted /workspace checkout and clone fresh.
FORCE_CLONE="${FORCE_CLONE:-0}"

clone_executorch() {
    echo "Cloning ExecuTorch ${EXECUTORCH_VERSION}..."
    rm -rf "${EXECUTORCH_DIR}"
    git clone --depth 1 --branch "${EXECUTORCH_VERSION}" \
        https://github.com/pytorch/executorch.git "${EXECUTORCH_DIR}"
}

# Prefer a volume-backed checkout under /workspace so rebuilds skip the full clone.
if [[ "${FORCE_CLONE}" == "1" ]]; then
    clone_executorch
elif [[ -d "${EXECUTORCH_DIR}/.git" ]]; then
    current_tag="$(git -C "${EXECUTORCH_DIR}" describe --tags --exact-match 2>/dev/null || true)"
    if [[ "${current_tag}" == "${EXECUTORCH_VERSION}" ]]; then
        echo "Reusing ExecuTorch ${EXECUTORCH_VERSION} at ${EXECUTORCH_DIR}"
        # Drop prior patch edits so apply-*.sh scripts stay idempotent and up to date.
        git -C "${EXECUTORCH_DIR}" reset --hard HEAD
    else
        echo "Persisted checkout is '${current_tag:-unknown}', want ${EXECUTORCH_VERSION}; re-cloning..."
        clone_executorch
    fi
else
    clone_executorch
fi

cd "${EXECUTORCH_DIR}"

echo "Initializing required submodules..."
git submodule sync
git -c advice.detachedHead=false submodule update --init --depth 1 --jobs "$(nproc)" \
    third-party/flatbuffers \
    third-party/flatcc \
    third-party/gflags \
    third-party/json \
    backends/xnnpack/third-party/FP16 \
    backends/xnnpack/third-party/FXdiv \
    backends/xnnpack/third-party/XNNPACK \
    backends/xnnpack/third-party/cpuinfo \
    backends/xnnpack/third-party/pthreadpool \
    backends/vulkan/third-party/Vulkan-Headers \
    backends/vulkan/third-party/VulkanMemoryAllocator \
    backends/vulkan/third-party/volk

echo "Patching ExecuTorch Vulkan backend for GPU input staging..."
bash /usr/local/share/infers/executorch-patches/apply-vulkan-gpu-input.sh "${EXECUTORCH_DIR}"

echo "Patching ExecuTorch Vulkan Runtime for external adapter teardown..."
bash /usr/local/share/infers/executorch-patches/apply-vulkan-external-adapter-teardown.sh "${EXECUTORCH_DIR}"

echo "Patching ExecuTorch SharedObject for GCC 15 <algorithm>..."
bash /usr/local/share/infers/executorch-patches/apply-vulkan-sharedobject-algorithm.sh "${EXECUTORCH_DIR}"

CMAKE_ARGS=(
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF
    -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=OFF
    -DEXECUTORCH_BUILD_PORTABLE_OPS=ON
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON
    -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON
    -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON
    -DEXECUTORCH_BUILD_XNNPACK=ON
    -DEXECUTORCH_BUILD_VULKAN=ON
    -DEXECUTORCH_ENABLE_PROGRAM_VERIFICATION=ON
    -DEXECUTORCH_ENABLE_LOGGING=ON
)

case "${TARGET}" in
    host)
        echo "Configuring ExecuTorch for host (${TARGET})..."
        # Ubuntu 26 / GCC 15: some ET third_party TUs omit <algorithm>.
        CMAKE_ARGS+=(-DCMAKE_CXX_FLAGS="-include algorithm")
        ;;
    android-arm64)
        echo "Configuring ExecuTorch for Android ${ANDROID_ABI} (API ${ANDROID_NDK_API})..."
        if [[ ! -d "${ANDROID_NDK_HOME}" ]]; then
            echo "ANDROID_NDK_HOME not found: ${ANDROID_NDK_HOME}" >&2
            exit 1
        fi
        if ! command -v glslc >/dev/null 2>&1; then
            echo "glslc from the Vulkan SDK must be on PATH (not NDK glslc)." >&2
            exit 1
        fi
        CMAKE_ARGS+=(
            -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake"
            -DANDROID_ABI="${ANDROID_ABI}"
            -DANDROID_PLATFORM="android-${ANDROID_NDK_API}"
        )
        ;;
    *)
        echo "Unknown TARGET=${TARGET}; expected host or android-arm64" >&2
        exit 1
        ;;
esac

echo "Configuring ExecuTorch..."
rm -rf "${BUILD_DIR}"
cmake "${CMAKE_ARGS[@]}" -B "${BUILD_DIR}" .

echo "Building ExecuTorch..."
cmake --build "${BUILD_DIR}" -j"$(nproc)"

echo "Copying static libraries to ${OUTPUT_DIR}..."
mkdir -p "${OUTPUT_DIR}"
# Replace prior contents so stale host/android objects cannot mix.
find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +

# Only ship the .a files cargo/executorch-sys link against. Skip CMake/ninja
# intermediates, object files, and vulkan_compute_shaders (~glsl/.spv).
REQUIRED_LIBS=(
    libexecutorch.a
    libexecutorch_core.a
    extension/data_loader/libextension_data_loader.a
    extension/flat_tensor/libextension_flat_tensor.a
    extension/module/libextension_module_static.a
    extension/named_data_map/libextension_named_data_map.a
    extension/tensor/libextension_tensor.a
    extension/threadpool/libextension_threadpool.a
    kernels/portable/libportable_ops_lib.a
    kernels/portable/libportable_kernels.a
    backends/xnnpack/libxnnpack_backend.a
    backends/xnnpack/third-party/pthreadpool/libpthreadpool.a
    backends/xnnpack/third-party/cpuinfo/libcpuinfo.a
    backends/xnnpack/third-party/XNNPACK/libXNNPACK.a
    backends/xnnpack/third-party/XNNPACK/libxnnpack-microkernels-prod.a
    backends/vulkan/libvulkan_backend.a
)
OPTIONAL_LIBS=(
    # Present on Android/arm64 XNNPACK builds (KleidiAI microkernels).
    kleidiai/libkleidiai.a
)

copy_lib() {
    local rel="$1"
    local src="${BUILD_DIR}/${rel}"
    local dst="${OUTPUT_DIR}/${rel}"
    if [[ ! -f "${src}" ]]; then
        echo "Missing required library: ${rel}" >&2
        return 1
    fi
    mkdir -p "$(dirname "${dst}")"
    cp -a "${src}" "${dst}"
}

for rel in "${REQUIRED_LIBS[@]}"; do
    copy_lib "${rel}"
done
for rel in "${OPTIONAL_LIBS[@]}"; do
    if [[ -f "${BUILD_DIR}/${rel}" ]]; then
        mkdir -p "$(dirname "${OUTPUT_DIR}/${rel}")"
        cp -a "${BUILD_DIR}/${rel}" "${OUTPUT_DIR}/${rel}"
    fi
done

chmod -R a+rwX "${OUTPUT_DIR}"

echo "ExecuTorch libraries built and copied to ${OUTPUT_DIR} (TARGET=${TARGET})"
