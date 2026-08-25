#!/usr/bin/env bash
set -euo pipefail

EXECUTORCH_VERSION="${EXECUTORCH_VERSION:-v1.4.0}"
EXECUTORCH_DIR="${EXECUTORCH_DIR:-/workspace/executorch}"
BUILD_DIR="${BUILD_DIR:-/workspace/build}"
OUTPUT_DIR="${OUTPUT_DIR:-/output}"

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

echo "Configuring ExecuTorch..."
rm -rf "${BUILD_DIR}"
cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF \
    -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=OFF \
    -DEXECUTORCH_BUILD_PORTABLE_OPS=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_VULKAN=ON \
    -DEXECUTORCH_ENABLE_PROGRAM_VERIFICATION=ON \
    -DEXECUTORCH_ENABLE_LOGGING=ON \
    -B "${BUILD_DIR}" \
    .

echo "Building ExecuTorch..."
cmake --build "${BUILD_DIR}" -j"$(nproc)"

echo "Copying build artifacts to ${OUTPUT_DIR}..."
mkdir -p "${OUTPUT_DIR}"
cp -a "${BUILD_DIR}/." "${OUTPUT_DIR}/"
chmod -R a+rwX "${OUTPUT_DIR}"

echo "ExecuTorch libraries built and copied to ${OUTPUT_DIR}"
