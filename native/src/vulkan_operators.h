#pragma once

#include "vulkan.h"

#include <cstdint>

namespace midas_native {

class VulkanOperators {
public:
    explicit VulkanOperators(VulkanContext& context);

    void conv(
        VulkanBuffer& output,
        const VulkanBuffer& input,
        const VulkanBuffer& weight,
        const VulkanBuffer& bias,
        std::uint32_t input_width,
        std::uint32_t input_height,
        std::uint32_t input_channels,
        std::uint32_t output_width,
        std::uint32_t output_height,
        std::uint32_t output_channels,
        std::uint32_t kernel_height,
        std::uint32_t kernel_width,
        std::uint32_t stride,
        std::int32_t padding_top,
        std::int32_t padding_left,
        std::uint32_t dilation,
        std::uint32_t groups,
        bool has_bias,
        const VulkanBuffer* gamma = nullptr,
        const VulkanBuffer* beta = nullptr,
        const VulkanBuffer* mean = nullptr,
        const VulkanBuffer* variance = nullptr,
        std::uint32_t activation = 0,
        bool relu_input = false,
        const VulkanBuffer* residual = nullptr);
    void batch_norm_activation(
        VulkanBuffer& output,
        const VulkanBuffer& input,
        const VulkanBuffer& gamma,
        const VulkanBuffer& beta,
        const VulkanBuffer& mean,
        const VulkanBuffer& variance,
        std::uint32_t count,
        std::uint32_t plane,
        std::uint32_t activation);
    void activation(
        VulkanBuffer& output,
        const VulkanBuffer& input,
        std::uint32_t count,
        std::uint32_t kind);
    void add(
        VulkanBuffer& output,
        const VulkanBuffer& left,
        const VulkanBuffer& right,
        std::uint32_t count);
    void resize(
        VulkanBuffer& output,
        const VulkanBuffer& input,
        std::uint32_t input_width,
        std::uint32_t input_height,
        std::uint32_t output_width,
        std::uint32_t output_height,
        std::uint32_t channels,
        bool align_corners);

private:
    VulkanContext& context_;
    VulkanPipeline conv_;
    VulkanPipeline conv_pointwise4_;
    VulkanPipeline conv_pointwise_gemm_;
    VulkanPipeline conv_pointwise_gemm_residual_;
    VulkanPipeline conv_depthwise3_;
    VulkanPipeline conv_spatial4_;
    VulkanPipeline conv_spatial4_tiled_;
    VulkanPipeline conv_spatial4_tiled_small_;
    VulkanPipeline conv_spatial4_tiled_relu_;
    VulkanPipeline batch_norm_activation_;
    VulkanPipeline activation_;
    VulkanPipeline add_;
    VulkanPipeline bilinear_;
};

}  // namespace midas_native
