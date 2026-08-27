#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname "$0")/.." && pwd)"

# Requires infers-android-builder (./scripts/docker_bake.sh android).
LOCAL_PROPS="$(mktemp)"
trap 'rm -f "${LOCAL_PROPS}"' EXIT
printf 'sdk.dir=/opt/android-sdk\n' > "${LOCAL_PROPS}"

# Forward publish / version env into the container when set.
# INFERS_VERSION also becomes VERSION_NAME for vanniktech (ORG_GRADLE_PROJECT_*).
docker_env_args=()
for var in \
    GITHUB_ACTOR \
    GITHUB_TOKEN \
    GITHUB_REPOSITORY \
    INFERS_VERSION
do
    if [[ -n "${!var:-}" ]]; then
        docker_env_args+=(-e "${var}")
    fi
done
if [[ -n "${INFERS_VERSION:-}" ]]; then
    docker_env_args+=(-e "ORG_GRADLE_PROJECT_VERSION_NAME=${INFERS_VERSION}")
fi

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
    "${docker_env_args[@]}" \
    infers-android-builder \
    "$@"
