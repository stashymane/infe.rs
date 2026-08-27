FROM ubuntu:26.04

ENV DEBIAN_FRONTEND=noninteractive

# Shared apt baseline for builder + android images.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    wget \
    unzip \
    git \
    build-essential \
    cmake \
    ninja-build \
    python3 \
    python3-pip \
    libvulkan-dev

WORKDIR /workspace
