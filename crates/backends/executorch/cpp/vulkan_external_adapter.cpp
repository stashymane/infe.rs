//! C ABI wrapper around ExecuTorch's Vulkan external adapter registration.
//!
//! Compiles against `vulkan_backend` symbols; include paths come from the
//! ExecuTorch source tree when present, otherwise a forward declaration is used.

#include <cstdint>
#include <vulkan/vulkan.h>

namespace vkcompute {
namespace vkapi {

class Adapter;

// Declared in executorch/backends/vulkan/runtime/vk_api/Runtime.h
Adapter* set_and_get_external_adapter(
    VkInstance instance = VK_NULL_HANDLE,
    VkPhysicalDevice physical_device = VK_NULL_HANDLE,
    VkDevice logical_device = VK_NULL_HANDLE);

} // namespace vkapi
} // namespace vkcompute

extern "C" {

// Returns non-null on success once an adapter has been registered (or already was).
void* infers_et_set_external_vulkan_adapter(
    void* instance,
    void* physical_device,
    void* device) {
  return static_cast<void*>(vkcompute::vkapi::set_and_get_external_adapter(
      reinterpret_cast<VkInstance>(instance),
      reinterpret_cast<VkPhysicalDevice>(physical_device),
      reinterpret_cast<VkDevice>(device)));
}

} // extern "C"
