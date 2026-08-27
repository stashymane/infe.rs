# syntax=docker/dockerfile:1
# Named context `infers-base` is provided by docker/docker-bake.hcl (target:base).
# Plain `docker build` works if a local `infers-base` image already exists.
FROM infers-base
ARG ANDROID_CMDLINE_TOOLS_VERSION=13114758
ARG ANDROID_PLATFORM=android-36
ARG ANDROID_BUILD_TOOLS=36.0.0
# NDK r27c (same family as infers-builder's side-by-side zip).
ARG ANDROID_NDK_VERSION=27.2.12479018
ARG RUST_TOOLCHAIN=nightly-2026-04-11

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    zip \
    clang \
    openjdk-17-jdk-headless \
    libssl-dev \
    pkg-config

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Android SDK cmdline-tools (zip cached under /var/cache/downloads).
ENV ANDROID_HOME=/opt/android-sdk
ENV ANDROID_SDK_ROOT=${ANDROID_HOME}
RUN --mount=type=cache,target=/var/cache/downloads,sharing=locked \
    set -eux; \
    ZIP="/var/cache/downloads/commandlinetools-linux-${ANDROID_CMDLINE_TOOLS_VERSION}_latest.zip"; \
    URL="https://dl.google.com/android/repository/commandlinetools-linux-${ANDROID_CMDLINE_TOOLS_VERSION}_latest.zip"; \
    if [ ! -s "${ZIP}" ]; then wget -q -O "${ZIP}.partial" "${URL}" && mv "${ZIP}.partial" "${ZIP}"; fi; \
    mkdir -p "${ANDROID_HOME}/cmdline-tools"; \
    unzip -q "${ZIP}" -d "${ANDROID_HOME}/cmdline-tools"; \
    mv "${ANDROID_HOME}/cmdline-tools/cmdline-tools" "${ANDROID_HOME}/cmdline-tools/latest"
ENV PATH="${ANDROID_HOME}/cmdline-tools/latest/bin:${ANDROID_HOME}/platform-tools:${PATH}"

# SDK packages + NDK (sdkmanager download cache survives rebuilds).
RUN --mount=type=cache,target=/root/.android/cache,sharing=locked \
    set -eux; \
    yes | sdkmanager --licenses >/dev/null; \
    sdkmanager --install \
        "platform-tools" \
        "platforms;${ANDROID_PLATFORM}" \
        "build-tools;${ANDROID_BUILD_TOOLS}" \
        "ndk;${ANDROID_NDK_VERSION}"; \
    ln -sfn "${ANDROID_HOME}/ndk/${ANDROID_NDK_VERSION}" /opt/android-ndk
ENV ANDROID_NDK_HOME=/opt/android-ndk
ENV PATH="${ANDROID_NDK_HOME}:${PATH}"

# Rust toolchain matching rust-toolchain.toml (+ Android target).
# rustup download cache under RUSTUP_HOME/downloads is Bind-mounted for rebuilds.
ENV RUSTUP_HOME=/opt/rustup
ENV CARGO_HOME=/opt/cargo
ENV PATH="${CARGO_HOME}/bin:${PATH}"
RUN --mount=type=cache,target=/opt/rustup/downloads,sharing=locked \
    --mount=type=cache,target=/opt/cargo/registry,sharing=locked \
    --mount=type=cache,target=/opt/cargo/git,sharing=locked \
    set -eux; \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none; \
    rustup toolchain install "${RUST_TOOLCHAIN}" \
        --component rust-src \
        --component rustc-dev \
        --component llvm-tools; \
    rustup default "${RUST_TOOLCHAIN}"; \
    rustup target add aarch64-linux-android --toolchain "${RUST_TOOLCHAIN}"

# Runtime caches survive image rebuilds when scripts/android_gradle.sh mounts
# the named volumes infers-android-gradle / infers-android-cargo-*.
VOLUME ["/root/.gradle", "/opt/cargo/registry", "/opt/cargo/git"]

# Entrypoint last so script edits do not bust toolchain layers.
COPY entrypoint.sh /usr/local/bin/android-entrypoint.sh
RUN chmod +x /usr/local/bin/android-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/android-entrypoint.sh"]
