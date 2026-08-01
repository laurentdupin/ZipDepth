#pragma once

#include "model.h"
#include "vulkan.h"
#include "vulkan_operators.h"
#include "zipdepth_ops.h"

#include <cstdint>
#include <string>
#include <unordered_map>

namespace zipdepth_native {

class VulkanExecutor {
public:
    struct Tensor {
        std::uint32_t channels=0,height=0,width=0;
        midas_native::VulkanBuffer buffer;
    };
    VulkanExecutor(const std::string& path, std::uint32_t device_index);
    zipdepth_model_kind kind() const { return model_.kind(); }
    midas_native::VulkanContext& context() { return context_; }
    const midas_native::VulkanContext& context() const { return context_; }
    Tensor infer_device(midas_native::VulkanBuffer input,
                        std::uint32_t width, std::uint32_t height);

private:
    const midas_native::VulkanBuffer& weight(const std::string& name) const;
    Tensor make(std::uint32_t c,std::uint32_t h,std::uint32_t w);
    Tensor conv(const Tensor& in,const std::string& weight,const char* bias,
                std::uint32_t stride=1,std::uint32_t padding=0,
                std::uint32_t dilation=1,std::uint32_t groups=1);
    Tensor bn(const Tensor& in,const std::string& prefix,bool relu);
    Tensor conv_bn(const Tensor& in,const std::string& prefix,
                   std::uint32_t stride=1,bool relu=true);
    Tensor rep(const Tensor& in,const std::string& prefix,
               std::uint32_t stride=1);
    Tensor add(const Tensor&a,const Tensor&b,float scale=1.0f);
    Tensor resize(const Tensor&in,std::uint32_t w,std::uint32_t h,
                  bool align_corners=false);
    Tensor nearest(const Tensor&in,std::uint32_t w,std::uint32_t h);
    Tensor adaptive(const Tensor&in,std::uint32_t w,std::uint32_t h);
    Tensor fusion(const Tensor& high,const Tensor& low,const std::string& prefix);

    ModelFile model_;
    midas_native::VulkanContext context_;
    midas_native::VulkanOperators operators_;
    ZipDepthOps extra_;
    midas_native::VulkanBuffer zero_;
    std::unordered_map<std::string,midas_native::VulkanBuffer> weights_;
};

}  // namespace zipdepth_native
