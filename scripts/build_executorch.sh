#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${ROOT}/target/executorch-libs"

mkdir -p "${OUTPUT_DIR}"

docker build -t executorch-builder -f "${ROOT}/docker/executorch.Dockerfile" "${ROOT}/docker"
docker run --rm -v "${OUTPUT_DIR}:/output" executorch-builder
