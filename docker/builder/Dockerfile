FROM ubuntu:26.04
ARG ANDROID_NDK_VERSION=r27c
ARG ANDROID_NDK_API=26
ARG VULKAN_SDK_VERSION=1.4.357.1
ARG TORCH_VERSION=2.13.0
ARG TORCHVISION_VERSION=0.28.0
ARG ULTRALYTICS_VERSION=8.4.128
ARG EXECUTORCH_PIP_VERSION=1.4.0

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    cmake \
    ninja-build \
    build-essential \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    wget \
    xz-utils \
    unzip \
    zlib1g-dev \
    libvulkan-dev \
    vulkan-tools \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxcb1 \
    libxext6 \
    libxrender1

# Host glslc from the Vulkan SDK (NDK glslc lacks GL_EXT_integer_dot_product).
RUN --mount=type=cache,target=/var/cache/downloads,sharing=locked \
    set -eux; \
    TAR="/var/cache/downloads/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.xz"; \
    URL="https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/linux/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.xz"; \
    if [ ! -s "${TAR}" ]; then wget -q -O "${TAR}.partial" "${URL}" && mv "${TAR}.partial" "${TAR}"; fi; \
    mkdir -p /opt/vulkan-sdk; \
    tar -xJf "${TAR}" -C /opt/vulkan-sdk --strip-components=1
ENV VULKAN_SDK=/opt/vulkan-sdk/x86_64
ENV PATH="${VULKAN_SDK}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${VULKAN_SDK}/lib"

# Android NDK for cross-compiling ExecuTorch static libs (arm64-v8a).
RUN --mount=type=cache,target=/var/cache/downloads,sharing=locked \
    set -eux; \
    ZIP="/var/cache/downloads/android-ndk-${ANDROID_NDK_VERSION}-linux.zip"; \
    URL="https://dl.google.com/android/repository/android-ndk-${ANDROID_NDK_VERSION}-linux.zip"; \
    if [ ! -s "${ZIP}" ]; then wget -q -O "${ZIP}.partial" "${URL}" && mv "${ZIP}.partial" "${ZIP}"; fi; \
    unzip -q "${ZIP}" -d /opt; \
    mv "/opt/android-ndk-${ANDROID_NDK_VERSION}" /opt/android-ndk
ENV ANDROID_NDK_HOME=/opt/android-ndk
ENV ANDROID_NDK_API=${ANDROID_NDK_API}
ENV PATH="${ANDROID_NDK_HOME}:${PATH}"

# Minimal Python env for ExecuTorch portable-ops / kernel codegen (needs torchgen + pyyaml).
# Avoids full ./install_requirements.sh (PyTorch AO, LLM tokenizers, lintrunner, …).
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    set -eux; \
    python3 -m venv /opt/et-venv; \
    /opt/et-venv/bin/pip install --upgrade pip; \
    /opt/et-venv/bin/pip install \
        "cmake>=3.24,<4.0.0" \
        pyyaml \
        packaging \
        "torch==${TORCH_VERSION}" \
        --extra-index-url https://download.pytorch.org/whl/cpu

# Separate env for YOLO26n-face → ExecuTorch .pte export (ultralytics + executorch pip).
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    set -eux; \
    python3 -m venv /opt/export-venv; \
    /opt/export-venv/bin/pip install --upgrade pip; \
    /opt/export-venv/bin/pip install \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        --extra-index-url https://download.pytorch.org/whl/cpu; \
    /opt/export-venv/bin/pip install \
        "ultralytics==${ULTRALYTICS_VERSION}" \
        "executorch==${EXECUTORCH_PIP_VERSION}" \
        pyyaml

WORKDIR /workspace

# Scripts/patches last so edits do not bust SDK/toolchain layers.
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY executorch-entrypoint.sh /usr/local/bin/executorch-entrypoint.sh
COPY patches /usr/local/share/infers/executorch-patches
COPY yolo26n-face/export.py /usr/local/share/infers/yolo26n-face/export.py
COPY yolo26n-face/entrypoint.sh /usr/local/bin/yolo26n-face-export-entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh \
    /usr/local/bin/executorch-entrypoint.sh \
    /usr/local/bin/yolo26n-face-export-entrypoint.sh \
    /usr/local/share/infers/executorch-patches/apply-vulkan-gpu-input.sh \
    /usr/local/share/infers/executorch-patches/apply-vulkan-external-adapter-teardown.sh \
    /usr/local/share/infers/executorch-patches/apply-vulkan-sharedobject-algorithm.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
