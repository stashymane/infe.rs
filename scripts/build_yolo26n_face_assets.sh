#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${ROOT}/assets/yolo26n-face"
IMGSZ="${IMGSZ:-192}"

mkdir -p "${OUTPUT_DIR}"

docker build -t yolo26n-face-exporter -f "${ROOT}/docker/yolo26n-face-export.Dockerfile" "${ROOT}/docker"

docker_run=(docker run --rm
    -e IMGSZ="${IMGSZ}"
    -e OUTPUT_DIR=/output
)

if [[ -n "${YOLO26N_FACE_WEIGHTS:-}" ]]; then
    docker_run+=(-e "YOLO26N_FACE_WEIGHTS=${YOLO26N_FACE_WEIGHTS}")
    docker_run+=(-v "${YOLO26N_FACE_WEIGHTS}:${YOLO26N_FACE_WEIGHTS}:ro")
fi

docker_run+=(-v "${OUTPUT_DIR}:/output" yolo26n-face-exporter)
"${docker_run[@]}"
