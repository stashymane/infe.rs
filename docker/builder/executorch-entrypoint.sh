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

echo "Cloning ExecuTorch ${EXECUTORCH_VERSION}..."
rm -rf "${EXECUTORCH_DIR}"
git clone --depth 1 --branch "${EXECUTORCH_VERSION}" \
    https://github.com/pytorch/executorch.git "${EXECUTORCH_DIR}"

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

echo "Copying build artifacts to ${OUTPUT_DIR}..."
mkdir -p "${OUTPUT_DIR}"
# Replace prior contents so stale host/android objects cannot mix.
find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
cp -a "${BUILD_DIR}/." "${OUTPUT_DIR}/"
chmod -R a+rwX "${OUTPUT_DIR}"

if [[ ! -f "${OUTPUT_DIR}/kernels/portable/libportable_ops_lib.a" ]]; then
    echo "Expected libportable_ops_lib.a missing under ${OUTPUT_DIR}" >&2
    exit 1
fi

echo "ExecuTorch libraries built and copied to ${OUTPUT_DIR} (TARGET=${TARGET})"
