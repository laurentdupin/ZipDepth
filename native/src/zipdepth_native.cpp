#include "zipdepth_native.h"
#include <inferbridge/linux_capture.h>

#include "cpu_executor.h"
#if defined(ZIPDEPTH_WITH_VULKAN)
#include "vulkan_executor.h"
#include "gpu_io.h"
#endif
#if defined(ZIPDEPTH_WITH_METAL)
#include "metal_executor.h"
#endif
#include "zipdepth_internal.h"
#include <inferbridge/native_harness_linux_dma_buf.h>
#include <inferbridge/native_harness_resource_cache.h>

#include <algorithm>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#if defined(__linux__) && !defined(__ANDROID__)
#include <atomic>
#include <chrono>
#include <cstdio>
#endif
#if defined(__ANDROID__)
#include <android/log.h>
#include <android/hardware_buffer.h>
#include <atomic>
#include <chrono>
#endif

#if defined(__linux__) && !defined(__ANDROID__) && defined(ZIPDEPTH_WITH_VULKAN)
struct ZipDepthDmaBufImage {
    midas_native::VulkanImage image;
};
#endif

#if defined(__ANDROID__) && defined(ZIPDEPTH_WITH_VULKAN)
struct ZipDepthAndroidImage {
    midas_native::VulkanImage image;
};
#endif

struct zipdepth_context {
    std::unique_ptr<zipdepth_native::CpuExecutor> cpu;
#if defined(ZIPDEPTH_WITH_VULKAN)
    std::unique_ptr<zipdepth_native::VulkanExecutor> gpu;
    std::unique_ptr<zipdepth_native::GpuIo> gpu_io;
#if defined(__linux__) && !defined(__ANDROID__)
    inferbridge::native_harness::StableResourceCache<ZipDepthDmaBufImage>
        dma_buf_images;
#endif
#endif
#if defined(ZIPDEPTH_WITH_METAL)
    std::unique_ptr<zipdepth_native::MetalExecutor> metal;
#endif
};

int ZIPDEPTH_CALL zipdepth_linux_capture_capabilities(
    zipdepth_context* context, ibr_linux_capture_capabilities* output) {
    if (!output) return IBRH_ERROR_INVALID_ARGUMENT;
    *output = {};
    output->struct_size = sizeof(*output);
#if defined(__linux__) && !defined(__ANDROID__) && defined(ZIPDEPTH_WITH_VULKAN)
    try {
        if (!context || !context->gpu) return IBRH_OK;
        auto& vk = context->gpu->context();
        if (!vk.external_capabilities().dma_buf_import) return IBRH_OK;
        const auto caps = inferbridge::linux_dma_buf::query(vk.physical_device(), VK_IMAGE_USAGE_SAMPLED_BIT);
        if (!caps.available()) return IBRH_OK;
        output->render_major = static_cast<uint32_t>(caps.render_major);
        output->render_minor = static_cast<uint32_t>(caps.render_minor);
        for (const auto& format : caps.formats) for (auto modifier : format.modifiers) {
            if (output->format_count == IBR_LINUX_CAPTURE_MAX_FORMATS) break;
            auto& entry = output->formats[output->format_count++];
            entry.pixel_format = format.format == VK_FORMAT_B8G8R8A8_UNORM ? IBRH_PIXEL_BGRA8 : IBRH_PIXEL_RGBA8;
            entry.modifier = modifier;
        }
    } catch (...) { output->format_count = 0; }
#else
    (void)context;
#endif
    return IBRH_OK;
}

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

namespace zipdepth_native {

#if defined(ZIPDEPTH_WITH_METAL)
class MetalContextExternalGpu final : public ExternalGpu {
public:
    explicit MetalContextExternalGpu(zipdepth_context* context)
        : context_(context) {}
    ExternalGpuCapabilities capabilities() const override {
        return {true, 0u, 3u};
    }
    std::shared_ptr<ExternalJob> submit_texture(
        const ExternalTextureRequest& request) override {
        if (context_ == nullptr || !context_->metal)
            throw std::invalid_argument("ZipDepth Metal context is unavailable");
        return context_->metal->submit_texture(request);
    }
    void transfer_counters(std::uint64_t& upload,
                           std::uint64_t& download) const override {
        upload = 0u;
        download = 0u;
    }
private:
    zipdepth_context* context_ = nullptr;
};
#endif

std::shared_ptr<ExternalGpu> create_metal_external_gpu(
    zipdepth_context* context) {
#if defined(ZIPDEPTH_WITH_METAL)
    if (context == nullptr || !context->metal)
        throw std::invalid_argument("ZipDepth Metal context is unavailable");
    return std::make_shared<MetalContextExternalGpu>(context);
#else
    (void)context;
    throw std::invalid_argument("ZipDepth was built without Metal");
#endif
}

}  // namespace zipdepth_native

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
        context->gpu_io = std::make_unique<zipdepth_native::GpuIo>(
            context->gpu->context());
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

zipdepth_status ZIPDEPTH_CALL zipdepth_create_metal(
    const char* model_path, zipdepth_context** output) {
    if (!output || !model_path || !model_path[0])
        return ZIPDEPTH_STATUS_INVALID_ARGUMENT;
    *output = nullptr;
#if !defined(ZIPDEPTH_WITH_METAL)
    return ZIPDEPTH_STATUS_UNSUPPORTED;
#else
    return protect([&] {
        auto context = std::make_unique<zipdepth_context>();
        context->cpu = std::make_unique<zipdepth_native::CpuExecutor>(model_path);
        context->metal = std::make_unique<zipdepth_native::MetalExecutor>(model_path);
        *output = context.release();
    });
#endif
}

zipdepth_status ZIPDEPTH_CALL zipdepth_infer_tensor_metal_f32(
    zipdepth_context* context, const float* rgb, uint32_t width,
    uint32_t height, float* depth, uint64_t elements) {
#if !defined(ZIPDEPTH_WITH_METAL)
    (void)context;(void)rgb;(void)width;(void)height;(void)depth;(void)elements;
    return ZIPDEPTH_STATUS_UNSUPPORTED;
#else
    if (!context || !context->metal || !rgb || !depth ||
        elements < std::uint64_t(width) * height)
        return ZIPDEPTH_STATUS_INVALID_ARGUMENT;
    return protect([&] {
        context->metal->infer(rgb, width, height, depth, elements);
    });
#endif
}

#if defined(__ANDROID__)
zipdepth_status ZIPDEPTH_CALL
zipdepth_infer_android_hardware_buffer_vulkan_f32(
    zipdepth_context* context, void* android_hardware_buffer,
    uint64_t hardware_buffer_id,
    int acquire_fence_fd, uint32_t source_width, uint32_t source_height,
    uint32_t network_width, uint32_t network_height, float* depth,
    uint64_t elements) {
#if !defined(ZIPDEPTH_WITH_VULKAN)
    (void)context;(void)android_hardware_buffer;(void)hardware_buffer_id;
    (void)acquire_fence_fd;
    (void)source_width;(void)source_height;(void)network_width;
    (void)network_height;(void)depth;(void)elements;
    return ZIPDEPTH_STATUS_VULKAN_UNAVAILABLE;
#else
    if (!context || !context->gpu || !context->gpu_io ||
        !android_hardware_buffer || !source_width || !source_height ||
        !network_width || !network_height || !depth ||
        elements < std::uint64_t(network_width) * network_height) {
        return ZIPDEPTH_STATUS_INVALID_ARGUMENT;
    }
    return protect([&] {
        using Clock = std::chrono::steady_clock;
        const auto started = Clock::now();
        auto& vk = context->gpu->context();
        const uint64_t image_id = hardware_buffer_id != 0u
            ? hardware_buffer_id
            : uint64_t(reinterpret_cast<uintptr_t>(android_hardware_buffer));
        (void)image_id;
        const bool cache_hit = false;
        // AImageReader may recycle an allocation as soon as both display and
        // inference release it. Qualcomm's imported-memory lifetime cannot be
        // cached safely across that recycling boundary, so keep the import
        // strictly inside the retained input frame's job.
        auto imported = std::make_unique<ZipDepthAndroidImage>();
        imported->image = vk.import_android_hardware_buffer(
            android_hardware_buffer, source_width, source_height,
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_SAMPLED_BIT |
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
        auto& image = imported->image;
        const auto imported_at = Clock::now();
        midas_native::VulkanSemaphore wait;
        if (acquire_fence_fd >= 0) {
            wait = vk.import_sync_fd(acquire_fence_fd);
        }
        auto input = vk.create_device_buffer(
            std::uint64_t(3u) * network_width * network_height * sizeof(float));
        midas_native::VulkanBuffer output;
        auto submission = vk.batch_async(
            std::move(wait), {}, [&] {
                vk.acquire_external_image(
                    image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_ACCESS_SHADER_READ_BIT);
                context->gpu_io->preprocess(
                    input, image, network_width, network_height);
                auto inferred = context->gpu->infer_device(
                    std::move(input), network_width, network_height);
                context->gpu_io->normalize_relative(
                    inferred.buffer, network_width * network_height);
                output = std::move(inferred.buffer);
                vk.release_external_image(
                    image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_ACCESS_SHADER_READ_BIT);
            });
        submission.wait();
        const auto inferred_at = Clock::now();
        vk.download(output, depth,
            static_cast<std::size_t>(
                std::uint64_t(network_width) * network_height * sizeof(float)));
        const auto finished = Clock::now();
        static std::atomic<uint64_t> count{0u};
        const uint64_t current = ++count;
        if (current <= 5u || current % 30u == 0u) {
            const auto milliseconds = [](auto begin, auto end) {
                return std::chrono::duration<double, std::milli>(
                    end - begin).count();
            };
            __android_log_print(ANDROID_LOG_INFO, "ZipDepthAHB",
                "frame=%llu cached=%d import=%.3f gpu=%.3f download=%.3f total=%.3f",
                static_cast<unsigned long long>(current), cache_hit ? 1 : 0,
                milliseconds(started, imported_at),
                milliseconds(imported_at, inferred_at),
                milliseconds(inferred_at, finished),
                milliseconds(started, finished));
        }
    });
#endif
}
#endif

#if defined(__linux__) && !defined(__ANDROID__)
zipdepth_status ZIPDEPTH_CALL zipdepth_infer_dma_buf_vulkan_f32(
    zipdepth_context* context, int dma_buf_fd, uint64_t allocation_size,
    uint64_t byte_offset, uint64_t drm_modifier,
    uint32_t source_row_stride, uint32_t source_width,
    uint32_t source_height, uint32_t source_is_rgba,
    int acquire_fence_fd, uint32_t network_width,
    uint32_t network_height, float* depth, uint64_t elements) {
#if !defined(ZIPDEPTH_WITH_VULKAN)
    (void)context;(void)dma_buf_fd;(void)allocation_size;(void)byte_offset;
    (void)drm_modifier;(void)source_row_stride;(void)source_width;
    (void)source_height;(void)source_is_rgba;(void)acquire_fence_fd;
    (void)network_width;(void)network_height;(void)depth;(void)elements;
    return ZIPDEPTH_STATUS_VULKAN_UNAVAILABLE;
#else
    const inferbridge::native_harness::LinuxDmaBufImage source{
        dma_buf_fd, allocation_size, byte_offset, drm_modifier,
        source_row_stride, source_width, source_height, source_is_rgba != 0u,
    };
    if (!context || !context->gpu || !context->gpu_io ||
        !inferbridge::native_harness::valid_linux_dma_buf_image(source) ||
        !network_width || !network_height || !depth ||
        elements < std::uint64_t(network_width) * network_height) {
        return ZIPDEPTH_STATUS_INVALID_ARGUMENT;
    }
    return protect([&] {
        using Clock = std::chrono::steady_clock;
        const auto started = Clock::now();
        auto& vk = context->gpu->context();
        const VkFormat format = source.rgba
            ? VK_FORMAT_R8G8B8A8_UNORM : VK_FORMAT_B8G8R8A8_UNORM;
        const auto usage = VK_IMAGE_USAGE_SAMPLED_BIT;
        const std::uint64_t identity =
            inferbridge::native_harness::linux_dma_buf_identity(dma_buf_fd);
        if (identity == 0)
            throw std::invalid_argument("could not identify DMA-BUF allocation");
        bool cache_hit = false;
        // This worker waits for each submission before processing the next;
        // eviction here cannot destroy an image still in flight.
        if (context->dma_buf_images.size() >= 16) context->dma_buf_images.clear();
        auto& imported = context->dma_buf_images.get_or_create(
            {identity, source_width, source_height, format, usage,
                allocation_size, byte_offset, source_row_stride, drm_modifier}, [&] {
                ZipDepthDmaBufImage created;
                created.image = vk.import_dma_buf(
                    dma_buf_fd, allocation_size, byte_offset, drm_modifier,
                    source_row_stride, source_width, source_height,
                    format, usage);
                return created;
            }, &cache_hit);
        auto& image = imported.image;
        const auto imported_at = Clock::now();
        midas_native::VulkanSemaphore wait;
        if (acquire_fence_fd >= 0) wait = vk.import_sync_fd(acquire_fence_fd);
        auto input = vk.create_device_buffer(
            std::uint64_t(3u) * network_width * network_height * sizeof(float));
        midas_native::VulkanBuffer output;
        auto submission = vk.batch_async(std::move(wait), {}, [&] {
            vk.acquire_external_image(
                image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_ACCESS_SHADER_READ_BIT);
            context->gpu_io->preprocess_capture(
                input, image, network_width, network_height);
            auto inferred = context->gpu->infer_device(
                std::move(input), network_width, network_height);
            context->gpu_io->normalize_relative(
                inferred.buffer, network_width * network_height);
            output = std::move(inferred.buffer);
            vk.release_external_image(
                image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_ACCESS_SHADER_READ_BIT);
        });
        submission.wait();
        const auto inferred_at = Clock::now();
        vk.download(output, depth, static_cast<std::size_t>(
            std::uint64_t(network_width) * network_height * sizeof(float)));
        const auto finished = Clock::now();
        static std::atomic<uint64_t> count{0u};
        const uint64_t current = ++count;
        if (current <= 5u || current % 60u == 0u) {
            const auto milliseconds = [](auto begin, auto end) {
                return std::chrono::duration<double, std::milli>(
                    end - begin).count();
            };
            std::fprintf(stderr,
                "ZipDepthDMA frame=%llu cached=%d import=%.3f gpu=%.3f download=%.3f total=%.3f\n",
                static_cast<unsigned long long>(current),
                cache_hit ? 1 : 0,
                milliseconds(started, imported_at),
                milliseconds(imported_at, inferred_at),
                milliseconds(inferred_at, finished),
                milliseconds(started, finished));
        }
    });
#endif
}
#endif

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
