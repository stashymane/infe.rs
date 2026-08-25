# Dockerfile for building ExecuTorch C++ runtime and extension static libraries
# following the official documentation:
# https://docs.pytorch.org/executorch/stable/using-executorch-building-from-source.html

FROM ubuntu:26.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    cmake \
    ninja-build \
    build-essential \
    python3 \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    curl \
    wget \
    xz-utils \
    zlib1g-dev \
    libvulkan-dev \
    vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

# LunarG Vulkan SDK provides a glslc that supports GL_EXT_integer_dot_product
# (required by ExecuTorch Vulkan; Ubuntu/NDK glslc is not sufficient).
ARG VULKAN_SDK_VERSION=1.4.321.1
RUN wget -q "https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/linux/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.xz" \
        -O /tmp/vulkansdk.tar.xz \
    && mkdir -p /opt/vulkan-sdk \
    && tar -xJf /tmp/vulkansdk.tar.xz -C /opt/vulkan-sdk --strip-components=1 \
    && rm /tmp/vulkansdk.tar.xz
ENV VULKAN_SDK=/opt/vulkan-sdk/x86_64
ENV PATH="${VULKAN_SDK}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${VULKAN_SDK}/lib"

# Clone ExecuTorch official release v1.4.0
ARG EXECUTORCH_VERSION=v1.4.0
RUN git clone --depth 1 --branch ${EXECUTORCH_VERSION} https://github.com/pytorch/executorch.git /workspace/executorch

WORKDIR /workspace/executorch

# Initialize submodules (including Vulkan third-party) and install requirements
RUN git submodule sync --recursive \
    && git submodule update --init --recursive \
    && git submodule update --init \
        backends/vulkan/third-party/Vulkan-Headers \
        backends/vulkan/third-party/volk \
        backends/vulkan/third-party/VulkanMemoryAllocator \
    && ./install_requirements.sh

# Build ExecuTorch C++ libraries with XNNPACK + Vulkan backends
RUN cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF \
    -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=OFF \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_VULKAN=ON \
    -DEXECUTORCH_ENABLE_PROGRAM_VERIFICATION=ON \
    -DEXECUTORCH_ENABLE_LOGGING=ON \
    -B/workspace/build .

RUN cmake --build /workspace/build -j$(nproc)

# Export the built static libraries to /output
CMD ["/bin/bash", "-c", "mkdir -p /output && cp -r /workspace/build/* /output/ && chmod -R 777 /output && echo 'ExecuTorch libraries built and copied to /output'"]
