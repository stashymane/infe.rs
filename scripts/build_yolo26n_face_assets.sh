#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${ROOT}/assets/yolo26n-face"
IMGSZ="${IMGSZ:-192}"

mkdir -p "${OUTPUT_DIR}"

export DOCKER_BUILDKIT=1
# Bake resolves local contexts relative to CWD (not the bake file).
(
    cd "${ROOT}/docker"
    docker buildx bake -f docker-bake.hcl --progress=plain builder
)

# Named volume keeps any workspace-side caches across rebuilds (shared with
# scripts/build_executorch.sh).
docker_run=(docker run --rm
    -e IMGSZ="${IMGSZ}"
    -e OUTPUT_DIR=/output
    -v infers-builder-workspace:/workspace
)

if [[ -n "${YOLO26N_FACE_WEIGHTS:-}" ]]; then
    docker_run+=(-e "YOLO26N_FACE_WEIGHTS=${YOLO26N_FACE_WEIGHTS}")
    docker_run+=(-v "${YOLO26N_FACE_WEIGHTS}:${YOLO26N_FACE_WEIGHTS}:ro")
fi

docker_run+=(-v "${OUTPUT_DIR}:/output" infers-builder yolo26n-face)
"${docker_run[@]}"
