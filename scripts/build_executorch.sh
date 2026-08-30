#!/usr/bin/env bash
# Materialize Nix-built ExecuTorch libs into target/executorch.
#
# Usage: ./scripts/build_executorch.sh [x86_64-unknown-linux-gnu|android-arm64]
set -euo pipefail

ROOT="$(cd -- "$(dirname "$0")/.." && pwd)"
TARGET="${1:-x86_64-unknown-linux-gnu}"

case "${TARGET}" in
    x86_64-unknown-linux-gnu | android-arm64) ;;
    *)
        echo "Usage: $0 [x86_64-unknown-linux-gnu|android-arm64]" >&2
        exit 1
        ;;
esac

OUT_DIR="${ROOT}/target/executorch"
OUT_LIBS="${OUT_DIR}/${TARGET}"

mkdir -p "${OUT_DIR}"
echo "Building executorch-${TARGET}..."
rm -rf "${OUT_LIBS}"
nix build "${ROOT}#executorch-${TARGET}" --out-link "${OUT_LIBS}"

echo "ExecuTorch ready at ${OUT_LIBS}"
