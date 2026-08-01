#pragma once

#include "vulkan.h"

#include <cstdint>

namespace zipdepth_native {

class ZipDepthOps {
public:
    explicit ZipDepthOps(midas_native::VulkanContext& context);
    void elementwise(midas_native::VulkanBuffer& out,
        const midas_native::VulkanBuffer& a,
        const midas_native::VulkanBuffer& b, std::uint32_t count,
        std::uint32_t plane, std::uint32_t op, float scale = 1.0f);
    void channel_average(midas_native::VulkanBuffer& out,
        const midas_native::VulkanBuffer& in, std::uint32_t channels,
        std::uint32_t plane);
    void strip_attention(midas_native::VulkanBuffer& out,
        const midas_native::VulkanBuffer& in,
        const midas_native::VulkanBuffer& weight,
        const midas_native::VulkanBuffer& gamma,
        const midas_native::VulkanBuffer& beta,
        const midas_native::VulkanBuffer& mean,
        const midas_native::VulkanBuffer& variance,
        std::uint32_t width, std::uint32_t height, std::uint32_t channels);
    void adaptive_average(midas_native::VulkanBuffer& out,
        const midas_native::VulkanBuffer& in, std::uint32_t iw,
        std::uint32_t ih, std::uint32_t ow, std::uint32_t oh,
        std::uint32_t channels);
    void nearest(midas_native::VulkanBuffer& out,
        const midas_native::VulkanBuffer& in, std::uint32_t iw,
        std::uint32_t ih, std::uint32_t ow, std::uint32_t oh,
        std::uint32_t channels);
    void maxpool5(midas_native::VulkanBuffer& out,
        const midas_native::VulkanBuffer& in, std::uint32_t width,
        std::uint32_t height, std::uint32_t channels);
    void concat4(midas_native::VulkanBuffer& out,
        const midas_native::VulkanBuffer& a,
        const midas_native::VulkanBuffer& b,
        const midas_native::VulkanBuffer& c,
        const midas_native::VulkanBuffer& d, std::uint32_t each_count);
    void context_reduce(midas_native::VulkanBuffer& out,
        const midas_native::VulkanBuffer& input,
        const midas_native::VulkanBuffer& logits,
        std::uint32_t channels, std::uint32_t plane);
    void convex(midas_native::VulkanBuffer& out,
        const midas_native::VulkanBuffer& depth,
        const midas_native::VulkanBuffer& weights,
        std::uint32_t width, std::uint32_t height);
    void mobile(midas_native::VulkanBuffer& out,
        const midas_native::VulkanBuffer& depth,
        const midas_native::VulkanBuffer& alpha,
        std::uint32_t width, std::uint32_t height);

private:
    midas_native::VulkanContext& context_;
    midas_native::VulkanPipeline elementwise_, channel_average_, strip_,
        adaptive_, nearest_, maxpool_, concat_, context_reduce_, convex_, mobile_;
};

}  // namespace zipdepth_native
