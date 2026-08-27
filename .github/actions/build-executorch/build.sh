#!/usr/bin/env bash
# Nix-build ExecuTorch for one target into target/executorch/.
#
# Copies store paths into the workspace so artifacts survive upload/download
# (out-links alone would break on downstream runners).
set -euo pipefail

ROOT="${GITHUB_WORKSPACE:-$(pwd)}"
TARGET="${1:?target}"

case "${TARGET}" in
    x86_64-unknown-linux-gnu | android-arm64) ;;
    *)
        echo "Unknown target: ${TARGET}" >&2
        exit 1
        ;;
esac

if ! command -v nix >/dev/null 2>&1; then
    echo "Nix is required (flakes enabled)." >&2
    exit 1
fi

cd "${ROOT}"
mkdir -p "${ROOT}/target/executorch"

out="${ROOT}/target/executorch/${TARGET}"
rm -rf "${out}"
nix build ".#executorch-${TARGET}" --out-link /tmp/et-libs
cp -a /tmp/et-libs/. "${out}"

if [[ "${TARGET}" == x86_64-unknown-linux-gnu ]]; then
    src="${ROOT}/target/executorch/executorch"
    rm -rf "${src}"
    nix build .#executorch-src --out-link /tmp/et-src
    cp -a /tmp/et-src/. "${src}"
fi

find "${ROOT}/target/executorch" -name '*.a' | head
