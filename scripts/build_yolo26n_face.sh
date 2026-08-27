#!/usr/bin/env bash
# Build yolo26n-face models into target/yolo26n-face (see assets/yolo26n-face-manifest.yaml).
set -euo pipefail

ROOT="$(cd -- "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

rm -rf target/yolo26n-face
nix build .#yolo26n-face-export --out-link target/yolo26n-face
