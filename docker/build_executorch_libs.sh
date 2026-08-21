#!/usr/bin/env bash
set -euo pipefail

if [ -n "${BASH_SOURCE[0]:-}" ]; then
    SCRIPT_PATH="${BASH_SOURCE[0]}"
else
    SCRIPT_PATH="$0"
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/target/executorch-libs"

echo "Building ExecuTorch libs..."
mkdir -p "${OUTPUT_DIR}"

docker build -t executorch-builder -f "${SCRIPT_DIR}/executorch.Dockerfile" "${SCRIPT_DIR}"
docker run --rm -v "${OUTPUT_DIR}:/output" executorch-builder

echo ""
echo "Build complete"
echo "Result output to ${OUTPUT_DIR}"
