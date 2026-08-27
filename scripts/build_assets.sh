#!/usr/bin/env bash
# Build Nix-produced project assets:
#   - Linux x86_64 ExecuTorch static libs → target/executorch/x86_64-unknown-linux-gnu
#   - Android arm64 ExecuTorch libs       → target/executorch/android-arm64
#   - yolo26n-face .pte models            → target/yolo26n-face
set -euo pipefail

ROOT="$(cd -- "$(dirname "$0")/.." && pwd)"

echo "==> ExecuTorch libraries (x86_64-unknown-linux-gnu)"
"${ROOT}/scripts/build_executorch.sh" x86_64-unknown-linux-gnu

echo "==> ExecuTorch libraries (android-arm64)"
"${ROOT}/scripts/build_executorch.sh" android-arm64

echo "==> yolo26n-face assets"
"${ROOT}/scripts/build_yolo26n_face.sh"

echo "All assets built."
echo "  ${ROOT}/target/executorch/x86_64-unknown-linux-gnu"
echo "  ${ROOT}/target/executorch/android-arm64"
echo "  ${ROOT}/target/yolo26n-face"
