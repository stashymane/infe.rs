#!/usr/bin/env bash
# Build all Docker-produced project assets:
#   - host ExecuTorch static libs      → target/executorch-libs
#   - Android arm64 ExecuTorch libs    → target/executorch-libs-android-arm64
#   - yolo26n-face .pte models         → assets/yolo26n-face
#
# Env (forwarded to the underlying scripts):
#   FORCE_CLONE, EXECUTORCH_VERSION, IMGSZ, YOLO26N_FACE_WEIGHTS
set -euo pipefail

ROOT="$(cd -- "$(dirname "$0")/.." && pwd)"

echo "==> Docker images (builder)"
"${ROOT}/scripts/docker_bake.sh" builder

echo "==> ExecuTorch libraries (host)"
"${ROOT}/scripts/build_executorch.sh" host

echo "==> ExecuTorch libraries (android-arm64)"
"${ROOT}/scripts/build_executorch.sh" android-arm64

echo "==> yolo26n-face assets"
"${ROOT}/scripts/build_yolo26n_face_assets.sh"

echo "All assets built."
echo "  ${ROOT}/target/executorch-libs"
echo "  ${ROOT}/target/executorch-libs-android-arm64"
echo "  ${ROOT}/assets/yolo26n-face"
