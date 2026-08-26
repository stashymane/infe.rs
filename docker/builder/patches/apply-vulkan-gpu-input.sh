#!/usr/bin/env bash
# Patches ExecuTorch VulkanBackend for GPU-direct input staging (Phase 1).
set -euo pipefail

VULKAN_BACKEND="${1:?executorch source dir}/backends/vulkan/runtime/VulkanBackend.cpp"

if [[ ! -f "${VULKAN_BACKEND}" ]]; then
    echo "VulkanBackend.cpp not found at ${VULKAN_BACKEND}" >&2
    exit 1
fi

if grep -q "infers_et_register_vulkan_graph" "${VULKAN_BACKEND}"; then
    echo "VulkanBackend already patched for GPU input staging"
    exit 0
fi

python3 - "${VULKAN_BACKEND}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()

extern_decl = """
extern "C" {
void infers_et_register_vulkan_graph(void* graph);
int infers_et_vulkan_should_skip_staging_copy(uint32_t input_index);
}

namespace executorch {
"""

needle = "namespace executorch {"
if needle not in text:
    raise SystemExit(f"patch anchor not found: {needle!r}")
text = text.replace(needle, extern_decl, 1)

register = """
    infers_et_register_vulkan_graph(compute_graph);
    return compute_graph;"""

if "    return compute_graph;" not in text:
    raise SystemExit("return compute_graph anchor not found")
text = text.replace("    return compute_graph;", register, 1)

old_copy = """        compute_graph->maybe_cast_and_copy_into_staging(
            compute_graph->inputs()[i].staging,
            args[i]->toTensor().const_data_ptr(),
            args[i]->toTensor().numel(),
            equivalent_scalar_type(args[i]->toTensor().scalar_type()));"""

new_copy = """        if (!infers_et_vulkan_should_skip_staging_copy(static_cast<uint32_t>(i))) {
          compute_graph->maybe_cast_and_copy_into_staging(
              compute_graph->inputs()[i].staging,
              args[i]->toTensor().const_data_ptr(),
              args[i]->toTensor().numel(),
              equivalent_scalar_type(args[i]->toTensor().scalar_type()));
        }"""

if old_copy not in text:
    raise SystemExit("staging copy anchor not found")
text = text.replace(old_copy, new_copy, 1)

path.write_text(text)
print(f"Patched {path}")
PY
