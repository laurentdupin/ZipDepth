#include "zipdepth_ops.h"

#include "adaptive_average_spv.h"
#include "channel_average_spv.h"
#include "concat4_spv.h"
#include "context_reduce_spv.h"
#include "convex_upsample_spv.h"
#include "elementwise_spv.h"
#include "maxpool5_spv.h"
#include "mobile_upsample_spv.h"
#include "nearest_spv.h"
#include "strip_attention_spv.h"

namespace zipdepth_native {
namespace {
std::uint32_t up(std::uint32_t n, std::uint32_t d) { return (n+d-1)/d; }
}

ZipDepthOps::ZipDepthOps(midas_native::VulkanContext& c)
    : context_(c),
      elementwise_(c.create_pipeline(midas_elementwise_spv,midas_elementwise_spv_size,3,16)),
      channel_average_(c.create_pipeline(midas_channel_average_spv,midas_channel_average_spv_size,2,8)),
      strip_(c.create_pipeline(midas_strip_attention_spv,midas_strip_attention_spv_size,7,12)),
      adaptive_(c.create_pipeline(midas_adaptive_average_spv,midas_adaptive_average_spv_size,2,20)),
      nearest_(c.create_pipeline(midas_nearest_spv,midas_nearest_spv_size,2,20)),
      maxpool_(c.create_pipeline(midas_maxpool5_spv,midas_maxpool5_spv_size,2,12)),
      concat_(c.create_pipeline(midas_concat4_spv,midas_concat4_spv_size,5,4)),
      context_reduce_(c.create_pipeline(midas_context_reduce_spv,midas_context_reduce_spv_size,3,8)),
      convex_(c.create_pipeline(midas_convex_upsample_spv,midas_convex_upsample_spv_size,3,8)),
      mobile_(c.create_pipeline(midas_mobile_upsample_spv,midas_mobile_upsample_spv_size,3,8)) {}

void ZipDepthOps::elementwise(midas_native::VulkanBuffer&o,const midas_native::VulkanBuffer&a,const midas_native::VulkanBuffer&b,std::uint32_t n,std::uint32_t plane,std::uint32_t op,float scale){struct P{std::uint32_t n,plane,op;float scale;}p{n,plane,op,scale};context_.dispatch(elementwise_,{&o,&a,&b},&p,sizeof(p),up(n,256));}
void ZipDepthOps::channel_average(midas_native::VulkanBuffer&o,const midas_native::VulkanBuffer&i,std::uint32_t c,std::uint32_t plane){struct P{std::uint32_t c,plane;}p{c,plane};context_.dispatch(channel_average_,{&o,&i},&p,sizeof(p),up(c,64));}
void ZipDepthOps::strip_attention(midas_native::VulkanBuffer&o,const midas_native::VulkanBuffer&i,const midas_native::VulkanBuffer&w,const midas_native::VulkanBuffer&g,const midas_native::VulkanBuffer&b,const midas_native::VulkanBuffer&m,const midas_native::VulkanBuffer&v,std::uint32_t width,std::uint32_t height,std::uint32_t c){struct P{std::uint32_t w,h,c;}p{width,height,c};context_.dispatch(strip_,{&o,&i,&w,&g,&b,&m,&v},&p,sizeof(p),up(width,8),up(height,8),c);}
void ZipDepthOps::adaptive_average(midas_native::VulkanBuffer&o,const midas_native::VulkanBuffer&i,std::uint32_t iw,std::uint32_t ih,std::uint32_t ow,std::uint32_t oh,std::uint32_t c){struct P{std::uint32_t iw,ih,ow,oh,c;}p{iw,ih,ow,oh,c};context_.dispatch(adaptive_,{&o,&i},&p,sizeof(p),up(ow,8),up(oh,8),c);}
void ZipDepthOps::nearest(midas_native::VulkanBuffer&o,const midas_native::VulkanBuffer&i,std::uint32_t iw,std::uint32_t ih,std::uint32_t ow,std::uint32_t oh,std::uint32_t c){struct P{std::uint32_t iw,ih,ow,oh,c;}p{iw,ih,ow,oh,c};context_.dispatch(nearest_,{&o,&i},&p,sizeof(p),up(ow,8),up(oh,8),c);}
void ZipDepthOps::maxpool5(midas_native::VulkanBuffer&o,const midas_native::VulkanBuffer&i,std::uint32_t w,std::uint32_t h,std::uint32_t c){struct P{std::uint32_t w,h,c;}p{w,h,c};context_.dispatch(maxpool_,{&o,&i},&p,sizeof(p),up(w,8),up(h,8),c);}
void ZipDepthOps::concat4(midas_native::VulkanBuffer&o,const midas_native::VulkanBuffer&a,const midas_native::VulkanBuffer&b,const midas_native::VulkanBuffer&c,const midas_native::VulkanBuffer&d,std::uint32_t n){context_.dispatch(concat_,{&o,&a,&b,&c,&d},&n,sizeof(n),up(n*4,256));}
void ZipDepthOps::context_reduce(midas_native::VulkanBuffer&o,const midas_native::VulkanBuffer&i,const midas_native::VulkanBuffer&l,std::uint32_t c,std::uint32_t plane){struct P{std::uint32_t c,plane;}p{c,plane};context_.dispatch(context_reduce_,{&o,&i,&l},&p,sizeof(p),up(c,64));}
void ZipDepthOps::convex(midas_native::VulkanBuffer&o,const midas_native::VulkanBuffer&d,const midas_native::VulkanBuffer&w,std::uint32_t width,std::uint32_t height){struct P{std::uint32_t w,h;}p{width,height};context_.dispatch(convex_,{&o,&d,&w},&p,sizeof(p),up(width*2,8),up(height*2,8));}
void ZipDepthOps::mobile(midas_native::VulkanBuffer&o,const midas_native::VulkanBuffer&d,const midas_native::VulkanBuffer&a,std::uint32_t width,std::uint32_t height){struct P{std::uint32_t w,h;}p{width,height};context_.dispatch(mobile_,{&o,&d,&a},&p,sizeof(p),up(width*2,8),up(height*2,8));}

}  // namespace zipdepth_native
