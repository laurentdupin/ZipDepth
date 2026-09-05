#pragma once
#include "vulkan.h"
#include <memory>
namespace zipdepth_native {
class GpuIo {
public:
 explicit GpuIo(midas_native::VulkanContext&);
 void preprocess(midas_native::VulkanBuffer&,const midas_native::VulkanImage&,std::uint32_t,std::uint32_t);
#if defined(__linux__) && !defined(__ANDROID__)
 void preprocess_capture(midas_native::VulkanBuffer&,const midas_native::VulkanImage&,std::uint32_t,std::uint32_t);
#endif
 void normalize_relative(midas_native::VulkanBuffer&,std::uint32_t);
 void resize_depth(midas_native::VulkanImage&,const midas_native::VulkanBuffer&,std::uint32_t,std::uint32_t);
private:
 midas_native::VulkanContext& context_;
#if defined(__linux__) && !defined(__ANDROID__)
 std::unique_ptr<midas_native::VulkanPipeline> capture_preprocess_;
#endif
 midas_native::VulkanPipeline preprocess_,resize_,reduce_minmax_,normalize_relative_;
};
}
