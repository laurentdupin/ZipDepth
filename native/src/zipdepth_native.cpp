#include "zipdepth_native.h"

#include "cpu_executor.h"
#if defined(ZIPDEPTH_WITH_VULKAN)
#include "vulkan_executor.h"
#endif

#include <algorithm>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>

struct zipdepth_context {
    std::unique_ptr<zipdepth_native::CpuExecutor> cpu;
#if defined(ZIPDEPTH_WITH_VULKAN)
    std::unique_ptr<zipdepth_native::VulkanExecutor> gpu;
#endif
};

namespace {
thread_local std::string g_error;

template<class Function>
zipdepth_status protect(Function&& function) {
    try {
        function();
        g_error.clear();
        return ZIPDEPTH_STATUS_OK;
    } catch (const std::bad_alloc&) {
        g_error = "out of memory";
        return ZIPDEPTH_STATUS_OUT_OF_MEMORY;
    } catch (const std::invalid_argument& error) {
        g_error = error.what();
        return ZIPDEPTH_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception& error) {
        g_error = error.what();
        return ZIPDEPTH_STATUS_INTERNAL_ERROR;
    }
}
}

extern "C" {

uint32_t ZIPDEPTH_CALL zipdepth_abi_version(void) {
    return ZIPDEPTH_ABI_VERSION;
}

const char* ZIPDEPTH_CALL zipdepth_last_error(void) {
    return g_error.c_str();
}

zipdepth_status ZIPDEPTH_CALL zipdepth_get_transfer_counters(
    zipdepth_transfer_counters* counters) {
    if (!counters || counters->struct_size < sizeof(*counters))
        return ZIPDEPTH_STATUS_INVALID_ARGUMENT;
    counters->api_version = ZIPDEPTH_ABI_VERSION;
#if defined(ZIPDEPTH_WITH_VULKAN)
    midas_native::global_transfer_counters(
        counters->tensor_upload_bytes, counters->tensor_download_bytes);
#else
    counters->tensor_upload_bytes = 0u;
    counters->tensor_download_bytes = 0u;
#endif
    return ZIPDEPTH_STATUS_OK;
}

zipdepth_status ZIPDEPTH_CALL zipdepth_create_vulkan(
    const char* model_path, uint32_t device_index,
    zipdepth_context** output) {
    if (!output || !model_path || !model_path[0])
        return ZIPDEPTH_STATUS_INVALID_ARGUMENT;
    *output = nullptr;
    return protect([&] {
        auto context = std::make_unique<zipdepth_context>();
        context->cpu = std::make_unique<zipdepth_native::CpuExecutor>(model_path);
#if defined(ZIPDEPTH_WITH_VULKAN)
        context->gpu = std::make_unique<zipdepth_native::VulkanExecutor>(
            model_path, device_index);
#else
        (void)device_index;
#endif
        *output = context.release();
    });
}

zipdepth_status ZIPDEPTH_CALL zipdepth_infer_tensor_vulkan_f32(
    zipdepth_context* context, const float* rgb, uint32_t width,
    uint32_t height, float* depth, uint64_t elements) {
#if !defined(ZIPDEPTH_WITH_VULKAN)
    (void)context;(void)rgb;(void)width;(void)height;(void)depth;(void)elements;
    return ZIPDEPTH_STATUS_VULKAN_UNAVAILABLE;
#else
    if (!context || !context->gpu || !rgb || !depth ||
        elements < std::uint64_t(width) * height)
        return ZIPDEPTH_STATUS_INVALID_ARGUMENT;
    return protect([&] {
        auto& vk = context->gpu->context();
        auto input = vk.create_device_buffer(
            std::uint64_t(3) * width * height * sizeof(float));
        vk.upload(input, rgb,
            static_cast<std::size_t>(std::uint64_t(3) * width * height * sizeof(float)));
        auto output = context->gpu->infer_device(std::move(input), width, height);
        vk.download(output.buffer, depth,
            static_cast<std::size_t>(std::uint64_t(width) * height * sizeof(float)));
    });
#endif
}

void ZIPDEPTH_CALL zipdepth_destroy(zipdepth_context* context) {
    delete context;
}

zipdepth_status ZIPDEPTH_CALL zipdepth_infer_rgb_f32(
    zipdepth_context* context, const float* rgb, uint32_t width,
    uint32_t height, float* depth, uint64_t elements) {
    if (!context || !context->cpu || !rgb || !depth ||
        elements < std::uint64_t(width) * height)
        return ZIPDEPTH_STATUS_INVALID_ARGUMENT;
    return protect([&] {
        const auto result = context->cpu->infer(rgb, width, height);
        std::copy(result.data.begin(), result.data.end(), depth);
    });
}

}
