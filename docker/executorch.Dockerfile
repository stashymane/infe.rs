FROM ubuntu:26.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
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
    zlib1g-dev \
    libvulkan-dev \
    vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

ARG VULKAN_SDK_VERSION=1.4.357.1
RUN wget -q "https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/linux/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.xz" \
        -O /tmp/vulkansdk.tar.xz \
    && mkdir -p /opt/vulkan-sdk \
    && tar -xJf /tmp/vulkansdk.tar.xz -C /opt/vulkan-sdk --strip-components=1 \
    && rm /tmp/vulkansdk.tar.xz
ENV VULKAN_SDK=/opt/vulkan-sdk/x86_64
ENV PATH="${VULKAN_SDK}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${VULKAN_SDK}/lib"

# Minimal Python env for portable-ops / kernel codegen (needs torchgen + pyyaml).
# Avoids full ./install_requirements.sh (PyTorch AO, LLM tokenizers, lintrunner, …).
RUN python3 -m venv /opt/et-venv \
    && /opt/et-venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/et-venv/bin/pip install --no-cache-dir \
        "cmake>=3.24,<4.0.0" \
        pyyaml \
        packaging \
        "torch==2.13.0" \
        --extra-index-url https://download.pytorch.org/whl/cpu
ENV PATH="/opt/et-venv/bin:${PATH}"

WORKDIR /workspace

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY patches /usr/local/share/infers/executorch-patches
RUN chmod +x /usr/local/bin/entrypoint.sh \
    /usr/local/share/infers/executorch-patches/apply-vulkan-gpu-input.sh \
    /usr/local/share/infers/executorch-patches/apply-vulkan-external-adapter-teardown.sh \
    /usr/local/share/infers/executorch-patches/apply-vulkan-sharedobject-algorithm.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
