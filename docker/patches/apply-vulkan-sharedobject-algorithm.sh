#!/usr/bin/env bash
# Fixes ExecuTorch SharedObject.cpp for libstdc++ that no longer transitively
# pulls in <algorithm> (seen with GCC 15 / Ubuntu 26.04).
set -euo pipefail

SHARED_OBJECT="${1:?executorch source dir}/backends/vulkan/runtime/graph/containers/SharedObject.cpp"

if [[ ! -f "${SHARED_OBJECT}" ]]; then
    echo "SharedObject.cpp not found at ${SHARED_OBJECT}" >&2
    exit 1
fi

if grep -q '#include <algorithm>' "${SHARED_OBJECT}"; then
    echo "SharedObject.cpp already includes <algorithm>"
    exit 0
fi

python3 - "${SHARED_OBJECT}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
needle = "#include <executorch/backends/vulkan/runtime/graph/containers/SharedObject.h>\n"
if needle not in text:
    raise SystemExit("SharedObject.cpp include anchor not found")
path.write_text(text.replace(needle, needle + "\n#include <algorithm>\n", 1))
print(f"Patched {path}")
PY
