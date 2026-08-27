#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname "$0")/.." && pwd)"
# host (default) | android-arm64
TARGET="${1:-host}"

case "${TARGET}" in
    host)
        OUTPUT_DIR="${ROOT}/target/executorch-libs"
        ;;
    android-arm64)
        OUTPUT_DIR="${ROOT}/target/executorch-libs-android-arm64"
        ;;
    *)
        echo "Usage: $0 [host|android-arm64]" >&2
        exit 1
        ;;
esac

mkdir -p "${OUTPUT_DIR}"

# Requires infers-builder (./scripts/docker_bake.sh builder). Named volume keeps
# ExecuTorch source, submodules, and per-target CMake trees.
# FORCE_CLONE=1 discards the persisted checkout and clones fresh.
# FORCE_BUILD=1 wipes the persisted CMake build dir for a clean rebuild.
docker run --rm \
    -e "TARGET=${TARGET}" \
    -e "FORCE_CLONE=${FORCE_CLONE:-0}" \
    -e "FORCE_BUILD=${FORCE_BUILD:-0}" \
    -e "EXECUTORCH_VERSION=${EXECUTORCH_VERSION:-v1.4.0}" \
    -v infers-builder-workspace:/workspace \
    -v "${OUTPUT_DIR}:/output" \
    infers-builder \
    executorch
