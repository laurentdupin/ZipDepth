#include "vulkan_operators.h"

#include "activation_spv.h"
#include "add_spv.h"
#include "batch_norm_activation_spv.h"
#include "bilinear_spv.h"
#include "conv2d_grouped_spv.h"
#include "conv2d_pointwise4_spv.h"
#include "conv2d_pointwise_gemm_spv.h"
#include "conv2d_pointwise_gemm_residual_spv.h"
#include "conv2d_depthwise3_spv.h"
#include "conv2d_spatial4_spv.h"
#include "conv2d_spatial4_stride2_direct_spv.h"
#include "conv2d_spatial4_tiled_spv.h"
#include "conv2d_spatial4_tiled_oc4_spv.h"
#include "conv2d_spatial4_stride2_tiled_spv.h"
#include "conv2d_spatial4_tiled_small_spv.h"
#include "conv2d_spatial4_tiled_relu_spv.h"

#include <stdexcept>
#include <cstdlib>
#include <vector>

namespace midas_native {
namespace {

std::uint32_t divide_up(std::uint32_t value, std::uint32_t divisor) {
    return (value + divisor - 1) / divisor;
}

bool environment_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

}  // namespace

VulkanOperators::VulkanOperators(VulkanContext& context)
    : context_(context),
      conv_(context.create_pipeline(
          midas_conv2d_grouped_spv,
          midas_conv2d_grouped_spv_size,
          4,
          68)),
      conv_pointwise4_(context.create_pipeline(
          midas_conv2d_pointwise4_spv,
          midas_conv2d_pointwise4_spv_size,
          4,
          68)),
      conv_pointwise_gemm_(context.create_pipeline(
          midas_conv2d_pointwise_gemm_spv,
          midas_conv2d_pointwise_gemm_spv_size,
          4,
          68)),
      conv_pointwise_gemm_residual_(context.create_pipeline(
          midas_conv2d_pointwise_gemm_residual_spv,
          midas_conv2d_pointwise_gemm_residual_spv_size,
          5,
          68)),
      conv_depthwise3_(context.create_pipeline(
          midas_conv2d_depthwise3_spv,
          midas_conv2d_depthwise3_spv_size,
          4,
          68)),
      conv_spatial4_(context.create_pipeline(
          midas_conv2d_spatial4_spv,
          midas_conv2d_spatial4_spv_size,
          4,
          68)),
      conv_spatial4_stride2_direct_(context.create_pipeline(
          midas_conv2d_spatial4_stride2_direct_spv,
          midas_conv2d_spatial4_stride2_direct_spv_size, 4, 68)),
      conv_spatial4_tiled_(context.create_pipeline(
          midas_conv2d_spatial4_tiled_spv,
          midas_conv2d_spatial4_tiled_spv_size,
          4,
          68)),
      conv_spatial4_tiled_oc4_(context.create_pipeline(
          midas_conv2d_spatial4_tiled_oc4_spv,
          midas_conv2d_spatial4_tiled_oc4_spv_size,
          4,
          68)),
      conv_spatial4_stride2_tiled_(context.create_pipeline(
          midas_conv2d_spatial4_stride2_tiled_spv,
          midas_conv2d_spatial4_stride2_tiled_spv_size,
          4,
          68)),
      conv_spatial4_tiled_small_(context.create_pipeline(
          midas_conv2d_spatial4_tiled_small_spv,
          midas_conv2d_spatial4_tiled_small_spv_size,
          4,
          68)),
      conv_spatial4_tiled_relu_(context.create_pipeline(
          midas_conv2d_spatial4_tiled_relu_spv,
          midas_conv2d_spatial4_tiled_relu_spv_size,
          4,
          68)),
      batch_norm_activation_(context.create_pipeline(
          midas_batch_norm_activation_spv,
          midas_batch_norm_activation_spv_size,
          6,
          16)),
      activation_(context.create_pipeline(
          midas_activation_spv,
          midas_activation_spv_size,
          2,
          8)),
      add_(context.create_pipeline(
          midas_add_spv,
          midas_add_spv_size,
          3,
          4)),
      bilinear_(context.create_pipeline(
          midas_bilinear_spv,
          midas_bilinear_spv_size,
          2,
          24)),
      use_spatial_tiled_oc4_((!context.is_discrete_gpu() ||
          environment_enabled("ZIPDEPTH_FORCE_SPATIAL_TILED_OC4")) &&
          !environment_enabled("ZIPDEPTH_DISABLE_SPATIAL_TILED_OC4")),
      enable_spatial_stride2_tiled_((context.is_discrete_gpu() ||
          environment_enabled("ZIPDEPTH_ENABLE_SPATIAL_STRIDE2_TILED")) &&
          !environment_enabled("ZIPDEPTH_DISABLE_SPATIAL_STRIDE2_TILED")) {
    conv_.set_debug_name("midas_conv2d_grouped");
    conv_pointwise4_.set_debug_name("midas_conv2d_pointwise4");
    conv_pointwise_gemm_.set_debug_name("midas_conv2d_pointwise_gemm");
    conv_pointwise_gemm_residual_.set_debug_name(
        "midas_conv2d_pointwise_gemm_residual");
    conv_depthwise3_.set_debug_name("midas_conv2d_depthwise3");
    conv_spatial4_.set_debug_name("midas_conv2d_spatial4");
    conv_spatial4_stride2_direct_.set_debug_name("midas_conv2d_spatial4_stride2_direct");
    const char* direct = std::getenv("ZIPDEPTH_ENABLE_STRIDE2_DIRECT");
    enable_spatial_stride2_direct_ = direct == nullptr || environment_enabled("ZIPDEPTH_ENABLE_STRIDE2_DIRECT");
    conv_spatial4_tiled_.set_debug_name(
        "midas_conv2d_spatial4_tiled");
    conv_spatial4_tiled_oc4_.set_debug_name(
        "midas_conv2d_spatial4_tiled_oc4");
    conv_spatial4_stride2_tiled_.set_debug_name(
        "midas_conv2d_spatial4_stride2_tiled");
    conv_spatial4_tiled_small_.set_debug_name(
        "midas_conv2d_spatial4_tiled_small");
    conv_spatial4_tiled_relu_.set_debug_name(
        "midas_conv2d_spatial4_tiled_relu");
    batch_norm_activation_.set_debug_name(
        "midas_batch_norm_activation");
    activation_.set_debug_name("midas_activation");
    add_.set_debug_name("midas_add");
    bilinear_.set_debug_name("midas_bilinear");
}

void VulkanOperators::conv(
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
    const VulkanBuffer* gamma,
    const VulkanBuffer* beta,
    const VulkanBuffer* mean,
    const VulkanBuffer* variance,
    std::uint32_t activation,
    bool relu_input,
    const VulkanBuffer* residual) {
    struct Parameters {
        std::uint32_t input_width;
        std::uint32_t input_height;
        std::uint32_t input_channels;
        std::uint32_t output_width;
        std::uint32_t output_height;
        std::uint32_t output_channels;
        std::uint32_t kernel_height;
        std::uint32_t kernel_width;
        std::uint32_t stride;
        std::int32_t padding_top;
        std::int32_t padding_left;
        std::uint32_t dilation;
        std::uint32_t groups;
        std::uint32_t has_bias;
        std::uint32_t has_batch_norm;
        std::uint32_t activation;
        float epsilon;
    } parameters{
        input_width, input_height, input_channels,
        output_width, output_height, output_channels,
        kernel_height, kernel_width, stride,
        padding_top, padding_left, dilation, groups, has_bias ? 1u : 0u,
        gamma != nullptr ? 1u : 0u, activation, 0.001f};
    const bool pointwise =
        groups == 1 && kernel_height == 1 && kernel_width == 1 &&
        stride == 1 && dilation == 1 && padding_top == 0 &&
        padding_left == 0 && input_width == output_width &&
        input_height == output_height;
    const bool depthwise =
        groups == input_channels && groups == output_channels &&
        kernel_height == 3 && kernel_width == 3 && stride <= 2 &&
        dilation == 1 && padding_top == 1 && padding_left == 1;
    const bool spatial4 =
        groups == 1 && kernel_height == 3 && kernel_width == 3 &&
        dilation == 1;
    const bool spatial4_tiled =
        spatial4 && stride == 1 && padding_top == 1 &&
        padding_left == 1 && input_width == output_width &&
        input_height == output_height;
    const bool spatial4_tiled_oc4 =
        spatial4_tiled && !relu_input && use_spatial_tiled_oc4_;
    const bool spatial4_stride2_tiled = spatial4 &&
        enable_spatial_stride2_tiled_ && stride == 2 &&
        padding_top == 1 && padding_left == 1;
    const bool spatial4_stride2_direct = spatial4 && enable_spatial_stride2_direct_ &&
        stride == 2 && padding_top == 1 && padding_left == 1 && output_channels % 4 == 0;
    const bool spatial4_tiled_small =
        spatial4_tiled && !relu_input &&
        output_width <= 8 && output_height <= 4;
    const bool pointwise_residual = pointwise && residual != nullptr;
    if (relu_input && !spatial4_tiled) {
        throw std::invalid_argument(
            "fused input ReLU requires tiled spatial convolution");
    }
    if (residual != nullptr && !pointwise) {
        throw std::invalid_argument(
            "fused residual requires pointwise convolution");
    }
    if (gamma != nullptr || beta != nullptr ||
        mean != nullptr || variance != nullptr) {
        throw std::invalid_argument(
            "convolution expects pre-folded batch normalization");
    }
    const VulkanPipeline& pipeline =
        pointwise
            ? (pointwise_residual
                ? conv_pointwise_gemm_residual_
                : conv_pointwise4_)
            :
        (depthwise ? conv_depthwise3_ :
        (spatial4_stride2_tiled
            ? conv_spatial4_stride2_tiled_
            : (spatial4_tiled
            ? (spatial4_tiled_small
                ? conv_spatial4_tiled_small_
                : spatial4_tiled_oc4
                ? conv_spatial4_tiled_oc4_
                : (relu_input
                ? conv_spatial4_tiled_relu_
                : conv_spatial4_tiled_))
            :
        (spatial4 ? (spatial4_stride2_direct ? conv_spatial4_stride2_direct_ : conv_spatial4_) : conv_))));
    std::vector<const VulkanBuffer*> resources{
        &output, &input, &weight, &bias};
    if (residual != nullptr) {
        resources.push_back(residual);
    }
    context_.dispatch(
        pipeline,
        resources,
        &parameters,
        sizeof(parameters),
        pointwise_residual
            ? divide_up(output_width * output_height, 64)
            : divide_up(output_width,
                (spatial4_tiled && !spatial4_tiled_small) ||
                    spatial4_stride2_tiled ? 16 : 8),
        pointwise_residual
            ? divide_up(output_channels, 64)
            : divide_up(output_height,
                spatial4_tiled || spatial4_stride2_tiled ? 4 : 8),
        pointwise_residual
            ? 1
            : pointwise
            ? divide_up(output_channels, 4)
            : (spatial4
                ? divide_up(output_channels,
                    spatial4_stride2_tiled ? 8 :
                    spatial4_tiled_oc4 ? 4 :
                    spatial4_tiled ? 8 : 4)
                : output_channels));
}

void VulkanOperators::batch_norm_activation(
    VulkanBuffer& output,
    const VulkanBuffer& input,
    const VulkanBuffer& gamma,
    const VulkanBuffer& beta,
    const VulkanBuffer& mean,
    const VulkanBuffer& variance,
    std::uint32_t count,
    std::uint32_t plane,
    std::uint32_t activation) {
    struct Parameters {
        std::uint32_t count;
        std::uint32_t plane;
        std::uint32_t activation;
        float epsilon;
    } parameters{count, plane, activation, 1.0e-5f};
    context_.dispatch(
        batch_norm_activation_,
        {&output, &input, &gamma, &beta, &mean, &variance},
        &parameters,
        sizeof(parameters),
        divide_up(count, 256));
}

void VulkanOperators::activation(
    VulkanBuffer& output,
    const VulkanBuffer& input,
    std::uint32_t count,
    std::uint32_t kind) {
    const std::uint32_t parameters[2] = {count, kind};
    context_.dispatch(
        activation_,
        {&output, &input},
        parameters,
        sizeof(parameters),
        divide_up(count, 256));
}

void VulkanOperators::add(
    VulkanBuffer& output,
    const VulkanBuffer& left,
    const VulkanBuffer& right,
    std::uint32_t count) {
    context_.dispatch(
        add_,
        {&output, &left, &right},
        &count,
        sizeof(count),
        divide_up(count, 256));
}

void VulkanOperators::resize(
    VulkanBuffer& output,
    const VulkanBuffer& input,
    std::uint32_t input_width,
    std::uint32_t input_height,
    std::uint32_t output_width,
    std::uint32_t output_height,
    std::uint32_t channels,
    bool align_corners) {
    const std::uint32_t parameters[6] = {
        input_width,
        input_height,
        output_width,
        output_height,
        channels,
        align_corners ? 1u : 0u};
    context_.dispatch(
        bilinear_,
        {&output, &input},
        parameters,
        sizeof(parameters),
        divide_up(output_width, 8),
        divide_up(output_height, 8),
        channels);
}

}  // namespace midas_native
