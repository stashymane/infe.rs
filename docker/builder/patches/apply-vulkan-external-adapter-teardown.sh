#!/usr/bin/env bash
# Patches ExecuTorch Vulkan Runtime to support tearing down the sticky external
# adapter while the caller's VkDevice is still alive.
set -euo pipefail

ET_ROOT="${1:?executorch source dir}"
RUNTIME_H="${ET_ROOT}/backends/vulkan/runtime/vk_api/Runtime.h"
RUNTIME_CPP="${ET_ROOT}/backends/vulkan/runtime/vk_api/Runtime.cpp"

if [[ ! -f "${RUNTIME_H}" || ! -f "${RUNTIME_CPP}" ]]; then
    echo "Runtime.h/cpp not found under ${ET_ROOT}" >&2
    exit 1
fi

if grep -q "clear_external_adapter" "${RUNTIME_H}"; then
    echo "Vulkan Runtime already patched for external adapter teardown"
    exit 0
fi

python3 - "${RUNTIME_H}" "${RUNTIME_CPP}" <<'PY'
from pathlib import Path
import sys

runtime_h = Path(sys.argv[1])
runtime_cpp = Path(sys.argv[2])

h = runtime_h.read_text()
decl_anchor = """Adapter* set_and_get_external_adapter(
    const VkInstance instance = VK_NULL_HANDLE,
    const VkPhysicalDevice physical_device = VK_NULL_HANDLE,
    const VkDevice logical_device = VK_NULL_HANDLE);
"""
decl_replacement = decl_anchor + """
// Destroy the sticky external adapter while the caller's VkDevice is still
// valid. Required for correct teardown when sharing an app-owned device.
void clear_external_adapter();
"""
if decl_anchor not in h:
    raise SystemExit("Runtime.h: set_and_get_external_adapter declaration not found")
runtime_h.write_text(h.replace(decl_anchor, decl_replacement, 1))
print(f"Patched {runtime_h}")

cpp = runtime_cpp.read_text()
old = """Adapter* set_and_get_external_adapter(
    const VkInstance instance,
    const VkPhysicalDevice physical_device,
    const VkDevice logical_device) {
  static std::mutex external_adapter_mutex;
  static std::unique_ptr<Adapter> p_external_adapter;

  const std::lock_guard<std::mutex> lock(external_adapter_mutex);
  if (!p_external_adapter) {
    p_external_adapter = init_external_adapter(
        instance,
        physical_device,
        logical_device,
        1,
        set_and_get_pipeline_cache_data_path(""));
  }

  return p_external_adapter.get();
}
"""
new = """namespace {

std::mutex& external_adapter_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::unique_ptr<Adapter>& external_adapter_slot() {
  static std::unique_ptr<Adapter> adapter;
  return adapter;
}

} // namespace

Adapter* set_and_get_external_adapter(
    const VkInstance instance,
    const VkPhysicalDevice physical_device,
    const VkDevice logical_device) {
  const std::lock_guard<std::mutex> lock(external_adapter_mutex());
  std::unique_ptr<Adapter>& p_external_adapter = external_adapter_slot();
  if (!p_external_adapter) {
    p_external_adapter = init_external_adapter(
        instance,
        physical_device,
        logical_device,
        1,
        set_and_get_pipeline_cache_data_path(""));
  }

  return p_external_adapter.get();
}

void clear_external_adapter() {
  const std::lock_guard<std::mutex> lock(external_adapter_mutex());
  external_adapter_slot().reset();
}
"""
if old not in cpp:
    raise SystemExit("Runtime.cpp: set_and_get_external_adapter body not found")
runtime_cpp.write_text(cpp.replace(old, new, 1))
print(f"Patched {runtime_cpp}")
PY
