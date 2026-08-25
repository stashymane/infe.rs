//! C ABI wrapper around ExecuTorch's Vulkan external adapter registration.
//!
//! Compiles against `vulkan_backend` symbols; include paths come from the
//! ExecuTorch source tree when present, otherwise a forward declaration is used.
//!
//! ExecuTorch's `set_and_get_external_adapter` constructs an `Adapter` /
//! `PhysicalDevice` without going through `runtime()`, which is the only path
//! that calls `volkInitialize` + `volkLoadInstance`. Without that, volk's
//! global Vulkan entry points remain null and the first
//! `vkGetPhysicalDeviceProperties` call segfaults.
//!
//! Teardown: call `infers_et_clear_external_vulkan_adapter` while the shared
//! `VkDevice` is still alive (after dropping all sessions that use it).

#include <cstdint>
#include <vulkan/vulkan.h>

// Volk is linked via `libvulkan_backend.a` (`USE_VULKAN_VOLK`).
extern "C" {
VkResult volkInitialize(void);
void volkLoadInstance(VkInstance instance);
void volkLoadDevice(VkDevice device);
}

namespace vkcompute {
namespace vkapi {

class Adapter;

// Declared in executorch/backends/vulkan/runtime/vk_api/Runtime.h
Adapter* set_and_get_external_adapter(
    VkInstance instance = VK_NULL_HANDLE,
    VkPhysicalDevice physical_device = VK_NULL_HANDLE,
    VkDevice logical_device = VK_NULL_HANDLE);

// Infers patch: destroy sticky external adapter while VkDevice is still valid.
void clear_external_adapter();

} // namespace vkapi
} // namespace vkcompute

extern "C" {

// Returns non-null on success once an adapter has been registered (or already was).
void* infers_et_set_external_vulkan_adapter(
    void* instance,
    void* physical_device,
    void* device) {
  auto vk_instance = reinterpret_cast<VkInstance>(instance);
  auto vk_physical = reinterpret_cast<VkPhysicalDevice>(physical_device);
  auto vk_device = reinterpret_cast<VkDevice>(device);

  if (vk_instance == VK_NULL_HANDLE || vk_physical == VK_NULL_HANDLE ||
      vk_device == VK_NULL_HANDLE) {
    return nullptr;
  }

  // Mirror `init_global_vulkan_runtime` + `create_instance` for the external path.
  if (volkInitialize() != VK_SUCCESS) {
    return nullptr;
  }
  volkLoadInstance(vk_instance);
  // External Adapter ctor never calls volkLoadDevice (owned-device path does).
  volkLoadDevice(vk_device);

  return static_cast<void*>(vkcompute::vkapi::set_and_get_external_adapter(
      vk_instance, vk_physical, vk_device));
}

void infers_et_clear_external_vulkan_adapter() {
  vkcompute::vkapi::clear_external_adapter();
}

} // extern "C"
