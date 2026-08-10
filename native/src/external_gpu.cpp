
#include "external_gpu.h"

#include "gpu_io.h"
#include "vulkan_executor.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include <utility>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#endif

namespace zipdepth_native {
namespace {

using midas_native::VulkanBuffer;
using midas_native::VulkanContext;
using midas_native::VulkanImage;
using midas_native::VulkanSemaphore;
using midas_native::VulkanSubmission;

struct NetworkShape {
    std::uint32_t width;
    std::uint32_t height;
};

NetworkShape network_shape(
    std::uint32_t source_width, std::uint32_t source_height,
    std::uint32_t shorter_side) {
    if (!source_width || !source_height || !shorter_side)
        throw std::invalid_argument("invalid ZipDepth network shape");
    const double scale = static_cast<double>(shorter_side) /
        static_cast<double>(std::min(source_width, source_height));
    const auto align32 = [](double value) {
        return std::max(32u, static_cast<std::uint32_t>(
            std::llround(value / 32.0) * 32.0));
    };
    return {align32(source_width * scale), align32(source_height * scale)};
}

#if defined(_WIN32)
using Microsoft::WRL::ComPtr;
constexpr std::uint32_t kMaxInFlightJobs = 3u;

void check_hresult(HRESULT result, const char* operation) {
    if (FAILED(result)) {
        throw std::runtime_error(
            std::string(operation) + " failed with HRESULT " +
            std::to_string(static_cast<long>(result)));
    }
}

ComPtr<ID3D12Device> matching_d3d12_device(std::uint64_t luid) {
    if (luid == 0u) return {};
    ComPtr<IDXGIFactory6> factory;
    check_hresult(
        CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)),
        "CreateDXGIFactory2");
    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> adapter;
        const HRESULT result = factory->EnumAdapters1(index, &adapter);
        if (result == DXGI_ERROR_NOT_FOUND) break;
        check_hresult(result, "EnumAdapters1");
        DXGI_ADAPTER_DESC1 description{};
        check_hresult(adapter->GetDesc1(&description), "GetDesc1");
        std::uint64_t candidate = 0u;
        std::memcpy(&candidate, &description.AdapterLuid, sizeof(candidate));
        if (candidate != luid) continue;
        ComPtr<ID3D12Device> device;
        check_hresult(
            D3D12CreateDevice(
                adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                IID_PPV_ARGS(&device)),
            "D3D12CreateDevice");
        return device;
    }
    return {};
}

void validate_texture(
    ID3D12Device* device, std::uintptr_t handle, std::uint32_t width,
    std::uint32_t height, DXGI_FORMAT format, const char* label) {
    ComPtr<ID3D12Resource> resource;
    check_hresult(device->OpenSharedHandle(
        reinterpret_cast<HANDLE>(handle), IID_PPV_ARGS(&resource)), label);
    const D3D12_RESOURCE_DESC d = resource->GetDesc();
    if (d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
        d.Width != width || d.Height != height ||
        d.DepthOrArraySize != 1u || d.MipLevels != 1u ||
        d.SampleDesc.Count != 1u || d.Format != format)
        throw std::invalid_argument("ZipDepth shared texture descriptor mismatch");
}

class ExternalJobImpl final : public ExternalJob {
public:
    ExternalJobImpl(std::shared_ptr<ExternalGpu> owner, VulkanImage input,
                    VulkanImage output, VulkanSubmission submission)
        : owner_(std::move(owner)), input_(std::move(input)),
          output_(std::move(output)), submission_(std::move(submission)) {}
    ~ExternalJobImpl() override {
        try { submission_.wait(); } catch (...) {}
        submission_ = {}; output_ = {}; input_ = {};
    }
    ExternalJobState state() const override {
        if (cancelled_.load()) return ExternalJobState::cancelled;
        return submission_.ready() ? ExternalJobState::complete :
                                     ExternalJobState::running;
    }
    void cancel() override { cancelled_.store(true); }
private:
    std::shared_ptr<ExternalGpu> owner_;
    VulkanImage input_;
    VulkanImage output_;
    VulkanSubmission submission_;
    std::atomic<bool> cancelled_{false};
};
#endif

class ExternalGpuImpl final : public ExternalGpu {
public:
    ExternalGpuImpl(const std::string& path, std::uint32_t index)
        : executor_(path, index), io_(executor_.context())
#if defined(_WIN32)
          , d3d12_(matching_d3d12_device(executor_.context().adapter_luid()))
#endif
          {}

    ExternalGpuCapabilities capabilities() const override {
#if defined(_WIN32)
        const auto& capabilities = executor_.context().external_capabilities();
        const bool available = d3d12_ != nullptr &&
            capabilities.d3d12_resource_import &&
            capabilities.d3d12_fence_import &&
            capabilities.d3d12_bgra8_sampled_image_import &&
            capabilities.d3d12_r32_storage_image_import;
        return {
            available,
            available ? executor_.context().adapter_luid() : 0u,
            available ? kMaxInFlightJobs : 0u};
#else
        return {};
#endif
    }

    std::shared_ptr<ExternalJob> submit_texture(
        const ExternalTextureRequest& request) override {
#if !defined(_WIN32)
        (void)request;
        throw std::runtime_error("ZipDepth D3D12 interop is unavailable");
#else
        if (!capabilities().available) {
            throw std::runtime_error(
                "complete ZipDepth D3D12/Vulkan interop is unavailable");
        }
        if (!request.shared_texture_handle || !request.wait_fence_handle ||
            !request.output_texture_handle || !request.signal_fence_handle ||
            !request.width || !request.height || !request.input_size ||
            request.output_width != request.width ||
            request.output_height != request.height) {
            throw std::invalid_argument("invalid ZipDepth GPU texture request");
        }
        validate_texture(d3d12_.Get(), request.shared_texture_handle,
            request.width, request.height,
            request.rgba ? DXGI_FORMAT_R8G8B8A8_UNORM :
                           DXGI_FORMAT_B8G8R8A8_UNORM,
            "OpenSharedHandle(ZipDepth input)");
        validate_texture(d3d12_.Get(), request.output_texture_handle,
            request.output_width, request.output_height,
            DXGI_FORMAT_R32_FLOAT, "OpenSharedHandle(ZipDepth output)");
        const NetworkShape shape = network_shape(
            request.width, request.height, request.input_size);
        try {
            std::lock_guard<std::mutex> lock(record_mutex_);
            VulkanContext& context = executor_.context();
            VulkanImage output = context.import_d3d12_image(
                reinterpret_cast<void*>(request.output_texture_handle),
                request.output_width, request.output_height,
                VK_FORMAT_R32_SFLOAT,
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
            VulkanImage input = context.import_d3d12_image(
                reinterpret_cast<void*>(request.shared_texture_handle),
                request.width, request.height,
                request.rgba ? VK_FORMAT_R8G8B8A8_UNORM :
                               VK_FORMAT_B8G8R8A8_UNORM,
                VK_IMAGE_USAGE_SAMPLED_BIT |
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
            VulkanSemaphore wait = context.import_d3d12_fence(
                reinterpret_cast<void*>(request.wait_fence_handle),
                request.wait_fence_value);
            VulkanSemaphore signal = context.import_d3d12_fence(
                reinterpret_cast<void*>(request.signal_fence_handle),
                request.signal_fence_value);
            VulkanSubmission submission = context.batch_async(
                std::move(wait), std::move(signal), [&] {
                    VulkanBuffer normalized = context.create_device_buffer(
                        static_cast<std::uint64_t>(shape.width) *
                        shape.height * 3u * sizeof(float));
                    context.acquire_external_image(
                        input, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_ACCESS_SHADER_READ_BIT);
                    context.acquire_external_image(
                        output, VK_IMAGE_LAYOUT_GENERAL,
                        VK_ACCESS_SHADER_WRITE_BIT);
                    io_.preprocess(
                        normalized, input,
                        static_cast<std::uint32_t>(shape.width),
                        static_cast<std::uint32_t>(shape.height));
                    auto depth = executor_.infer_device(
                        std::move(normalized),
                        static_cast<std::uint32_t>(shape.width),
                        static_cast<std::uint32_t>(shape.height));
                    // ZipDepth is affine-invariant inverse depth. Keep the GPU
                    // publication identical to the host ABI path: normalize
                    // the complete neural output before resizing it into the
                    // caller-owned source-resolution texture.
                    io_.normalize_relative(
                        depth.buffer,
                        static_cast<std::uint32_t>(shape.width * shape.height));
                    io_.resize_depth(
                        output, depth.buffer,
                        static_cast<std::uint32_t>(shape.width),
                        static_cast<std::uint32_t>(shape.height));
                    context.release_external_image(
                        input, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_ACCESS_SHADER_READ_BIT);
                    context.release_external_image(
                        output, VK_IMAGE_LAYOUT_GENERAL,
                        VK_ACCESS_SHADER_WRITE_BIT);
                });
            return std::make_shared<ExternalJobImpl>(
                shared_from_this(), std::move(input), std::move(output),
                std::move(submission));
        } catch (...) { throw; }
#endif
    }

    void transfer_counters(
        std::uint64_t& upload_bytes,
        std::uint64_t& download_bytes) const override {
        executor_.context().transfer_counters(upload_bytes, download_bytes);
    }

private:
    VulkanExecutor executor_;
    GpuIo io_;
#if defined(_WIN32)
    ComPtr<ID3D12Device> d3d12_;
    std::mutex record_mutex_;
#endif
};

}  // namespace

std::shared_ptr<ExternalGpu> create_external_gpu(
    const std::string& path,
    std::uint32_t index) {
    return std::make_shared<ExternalGpuImpl>(path, index);
}

ExternalGpuCapabilities probe_external_gpu(std::uint32_t index) {
#if defined(_WIN32)
    VulkanContext context(index);
    const auto device = matching_d3d12_device(context.adapter_luid());
    const auto& capabilities = context.external_capabilities();
    const bool available = device != nullptr &&
        capabilities.d3d12_resource_import &&
        capabilities.d3d12_fence_import &&
        capabilities.d3d12_bgra8_sampled_image_import &&
        capabilities.d3d12_r32_storage_image_import;
    return {
        available,
        available ? context.adapter_luid() : 0u,
        available ? kMaxInFlightJobs : 0u};
#else
    (void)index;
    return {};
#endif
}

}  // namespace zipdepth_native
