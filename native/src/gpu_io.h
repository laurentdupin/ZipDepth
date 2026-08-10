#pragma once
#include "vulkan.h"
namespace zipdepth_native {
class GpuIo {
public:
 explicit GpuIo(midas_native::VulkanContext&);
 void preprocess(midas_native::VulkanBuffer&,const midas_native::VulkanImage&,std::uint32_t,std::uint32_t);
 void normalize_relative(midas_native::VulkanBuffer&,std::uint32_t);
 void resize_depth(midas_native::VulkanImage&,const midas_native::VulkanBuffer&,std::uint32_t,std::uint32_t);
private:
 midas_native::VulkanContext& context_;
 midas_native::VulkanPipeline preprocess_,resize_,reduce_minmax_,normalize_relative_;
};
}
