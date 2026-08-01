#include "gpu_io.h"
#include "preprocess_texture_spv.h"
#include "resize_depth_image_spv.h"
#include <stdexcept>
namespace zipdepth_native {
GpuIo::GpuIo(midas_native::VulkanContext&c):context_(c),preprocess_(c.create_pipeline(midas_preprocess_texture_spv,midas_preprocess_texture_spv_size,{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},{VK_ACCESS_SHADER_READ_BIT,VK_ACCESS_SHADER_WRITE_BIT},16)),resize_(c.create_pipeline(midas_resize_depth_image_spv,midas_resize_depth_image_spv_size,{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},{VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT},16)){}
void GpuIo::preprocess(midas_native::VulkanBuffer&o,const midas_native::VulkanImage&i,std::uint32_t w,std::uint32_t h){if(!w||!h||o.size()<std::uint64_t(w)*h*3*sizeof(float))throw std::invalid_argument("invalid ZipDepth preprocess shape");std::uint32_t p[4]={i.width(),i.height(),w,h};context_.dispatch_image_to_buffer(preprocess_,i,o,p,sizeof(p),(w+7)/8,(h+7)/8);}
void GpuIo::resize_depth(midas_native::VulkanImage&o,const midas_native::VulkanBuffer&i,std::uint32_t w,std::uint32_t h){std::uint32_t p[4]={w,h,o.width(),o.height()};context_.dispatch_buffer_to_image(resize_,i,o,p,sizeof(p),(o.width()+7)/8,(o.height()+7)/8);}
}
