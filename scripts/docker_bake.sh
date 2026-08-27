#!/usr/bin/env bash
# Bake local Docker images for development.
# CI builds the same targets via docker/bake-action (see publish-kotlin.yml).
#
# Usage (from repo root or anywhere):
#   ./scripts/docker_bake.sh              # default group: base, builder, android
#   ./scripts/docker_bake.sh builder
#   ./scripts/docker_bake.sh android
set -euo pipefail

ROOT="$(cd -- "$(dirname "$0")/.." && pwd)"

export DOCKER_BUILDKIT=1
# Bake resolves local contexts relative to CWD (not the bake file).
(
    cd "${ROOT}/docker"
    docker buildx bake -f docker-bake.hcl --progress=plain "$@"
)
