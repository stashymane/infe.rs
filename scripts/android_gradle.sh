#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname "$0")/.." && pwd)"
IMAGE="${INFERS_ANDROID_IMAGE:-infers-android-builder}"

export DOCKER_BUILDKIT=1

docker build \
    --progress=plain \
    -t "${IMAGE}" \
    -f "${ROOT}/docker/android/Dockerfile" \
    "${ROOT}/docker/android"

LOCAL_PROPS="$(mktemp)"
trap 'rm -f "${LOCAL_PROPS}"' EXIT
printf 'sdk.dir=/opt/android-sdk\n' > "${LOCAL_PROPS}"

# Persistent named volumes: Gradle + Cargo downloads survive container rebuilds.
# --network host: wireless/USB adb on the host is visible inside the container.
# Overlay local.properties so the host sdk.dir is not rewritten.
docker run --rm --network host \
    -v "${ROOT}:/workspace" \
    -v "${LOCAL_PROPS}:/workspace/kotlin/local.properties:ro" \
    -v "${HOME}/.android:/root/.android" \
    -v infers-android-gradle:/root/.gradle \
    -v infers-android-cargo-registry:/opt/cargo/registry \
    -v infers-android-cargo-git:/opt/cargo/git \
    -e GRADLE_USER_HOME=/root/.gradle \
    -e CARGO_HOME=/opt/cargo \
    -e CARGO_TARGET_DIR=/workspace/target \
    -e RUSTUP_HOME=/opt/rustup \
    "${IMAGE}" \
    "$@"
