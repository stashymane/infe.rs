//! GPU input staging helpers for ExecuTorch Vulkan delegate integration.
//!
//! When linked against a patched `libvulkan_backend.a` (see
//! `docker/builder/patches/apply-vulkan-gpu-input.sh`), the Vulkan backend registers its
//! `ComputeGraph` at init and honors a per-input skip mask after GPU copies into
//! delegate staging buffers.

#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.h>

#if defined(INFERS_ET_EXECUTORCH_SRC)
#include <executorch/backends/vulkan/runtime/api/containers/StagingBuffer.h>
#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>
#endif

namespace {

std::mutex g_graph_mutex;
void* g_registered_graph = nullptr;

thread_local uint64_t g_skip_staging_copy_mask = 0;

} // namespace

extern "C" {

void infers_et_register_vulkan_graph(void* graph) {
    std::lock_guard<std::mutex> lock(g_graph_mutex);
    g_registered_graph = graph;
}

void* infers_et_take_registered_vulkan_graph(void) {
    std::lock_guard<std::mutex> lock(g_graph_mutex);
    void* graph = g_registered_graph;
    g_registered_graph = nullptr;
    return graph;
}

void infers_et_vulkan_set_skip_staging_copy_mask(uint64_t mask) {
    g_skip_staging_copy_mask = mask;
}

int infers_et_vulkan_should_skip_staging_copy(uint32_t input_index) {
    if (input_index >= 64) {
        return 0;
    }
    return (g_skip_staging_copy_mask >> input_index) & 1u;
}

int infers_et_vulkan_input_staging_buffer(
    void* graph_ptr,
    uint32_t input_index,
    VkBuffer* out_buffer,
    VkDeviceSize* out_offset,
    VkDeviceSize* out_size) {
#if defined(INFERS_ET_EXECUTORCH_SRC)
    if (graph_ptr == nullptr || out_buffer == nullptr || out_offset == nullptr ||
        out_size == nullptr) {
        return -1;
    }

    auto* graph = static_cast<vkcompute::ComputeGraph*>(graph_ptr);
    if (input_index >= graph->inputs().size()) {
        return -2;
    }

    const vkcompute::ValueRef staging_ref = graph->inputs()[input_index].staging;
    if (!vkcompute::is_valid(staging_ref)) {
        return -3;
    }

    vkcompute::StagingPtr staging_ptr(graph, staging_ref);
    vkcompute::api::StagingBuffer& staging = *staging_ptr;
    vkcompute::vkapi::VulkanBuffer& dst = staging.buffer();

    *out_buffer = dst.handle();
    *out_offset = dst.mem_offset();
    *out_size = dst.mem_size();
    return 0;
#else
    (void)graph_ptr;
    (void)input_index;
    (void)out_buffer;
    (void)out_offset;
    (void)out_size;
    return -100;
#endif
}

} // extern "C"
