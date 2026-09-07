#pragma once

#include "vulkan.h"
#include <inferbridge/native_harness_int8_workspace.h>

#include <cstdint>

namespace zipdepth_native {

class ZipDepthOps {
public:
    explicit ZipDepthOps(midas_native::VulkanContext& context);
    void decoder_fusion(midas_native::VulkanBuffer& out,
        const midas_native::VulkanBuffer& low, const midas_native::VulkanBuffer& high,
        const midas_native::VulkanBuffer& gamma, const midas_native::VulkanBuffer& beta,
        const midas_native::VulkanBuffer& mean, const midas_native::VulkanBuffer& variance,
        std::uint32_t iw, std::uint32_t ih, std::uint32_t ow, std::uint32_t oh, std::uint32_t channels);
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
    void spatial_int8(midas_native::VulkanBuffer& out,
        const midas_native::VulkanBuffer& input,
        const midas_native::VulkanBuffer& packed_weight,
        const midas_native::VulkanBuffer& weight_scale,
        const midas_native::VulkanBuffer& bias,
        std::uint32_t width, std::uint32_t height,
        std::uint32_t input_channels, std::uint32_t output_channels,
        bool has_bias, bool relu = false);
    void pointwise_fp16_weights(midas_native::VulkanBuffer& out,
        const midas_native::VulkanBuffer& input,
        const midas_native::VulkanBuffer& weight,
        const midas_native::VulkanBuffer& bias,
        std::uint32_t spatial, std::uint32_t input_channels,
        std::uint32_t output_channels, bool has_bias);
private:
    midas_native::VulkanContext& context_;
    midas_native::VulkanPipeline decoder_fusion_;
    inferbridge::native::Int8ActivationWorkspace<midas_native::VulkanBuffer>
        int8_workspace_;
    midas_native::VulkanPipeline elementwise_, channel_average_, strip_,
        adaptive_, nearest_, maxpool_, concat_, context_reduce_, convex_, mobile_,
        reduce_absmax_, quantize_int8_, spatial_int8_, spatial_int8_relu_,
        pointwise_fp16_weights_;
};

}  // namespace zipdepth_native
