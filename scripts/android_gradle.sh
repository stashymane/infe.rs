#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname "$0")/.." && pwd)"

export DOCKER_BUILDKIT=1
# Bake resolves local contexts relative to CWD (not the bake file).
(
    cd "${ROOT}/docker"
    docker buildx bake -f docker-bake.hcl --progress=plain android
)

LOCAL_PROPS="$(mktemp)"
trap 'rm -f "${LOCAL_PROPS}"' EXIT
printf 'sdk.dir=/opt/android-sdk\n' > "${LOCAL_PROPS}"

# Persistent named volumes: Gradle + Cargo downloads survive container rebuilds.
# --network host: wireless/USB adb on the host is visible inside the container.
# Overlay local.properties so the host sdk.dir is not rewritten.
# HOST_UID/GID: chown workspace outputs back to the invoking user (IDE sync).
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
    -e "HOST_UID=$(id -u)" \
    -e "HOST_GID=$(id -g)" \
    infers-android-builder \
    "$@"
