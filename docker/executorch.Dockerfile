# Dockerfile for building ExecuTorch C++ runtime and extension static libraries
# following the official documentation:
# https://docs.pytorch.org/executorch/stable/using-executorch-building-from-source.html

FROM ubuntu:22.04

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
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone ExecuTorch official release v1.4.0
ARG EXECUTORCH_VERSION=v1.4.0
RUN git clone --depth 1 --branch ${EXECUTORCH_VERSION} https://github.com/pytorch/executorch.git /workspace/executorch

WORKDIR /workspace/executorch

# Initialize submodules and install requirements according to official instructions
RUN git submodule sync --recursive \
    && git submodule update --init --recursive \
    && ./install_requirements.sh

# Build ExecuTorch C++ libraries with standard extensions
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
    -DEXECUTORCH_ENABLE_PROGRAM_VERIFICATION=ON \
    -DEXECUTORCH_ENABLE_LOGGING=ON \
    -B/workspace/build .

RUN cmake --build /workspace/build -j$(nproc)

# Export the built static libraries to /output
CMD ["/bin/bash", "-c", "mkdir -p /output && cp -r /workspace/build/* /output/ && chmod -R 777 /output && echo 'ExecuTorch libraries built and copied to /output'"]
