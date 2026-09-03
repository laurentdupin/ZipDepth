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
#include "quantize_nchw_int8_spv.h"
#include "reduce_absmax_spv.h"
#include "conv2d_spatial_int8_spv.h"
#include "conv2d_spatial_int8_relu_spv.h"
#include "strip_attention_spv.h"
#include "precision_pointwise_fp16_weights_spv.h"

#include <algorithm>

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
      mobile_(c.create_pipeline(midas_mobile_upsample_spv,midas_mobile_upsample_spv_size,3,8)) {
    if (c.supports_float16()) {
        pointwise_fp16_weights_ = c.create_pipeline(
            midas_precision_pointwise_fp16_weights_spv,
            midas_precision_pointwise_fp16_weights_spv_size, 4, 16);
        pointwise_fp16_weights_.set_debug_name("zipdepth_pointwise_fp16_weights");
    }
    if (c.supports_packed_int8_dot()) {
        reduce_absmax_=c.create_pipeline(midas_reduce_absmax_spv,midas_reduce_absmax_spv_size,2,8);
        quantize_int8_=c.create_pipeline(midas_quantize_nchw_int8_spv,midas_quantize_nchw_int8_spv_size,3,12);
        spatial_int8_=c.create_pipeline(midas_conv2d_spatial_int8_spv,midas_conv2d_spatial_int8_spv_size,6,20);
        spatial_int8_relu_=c.create_pipeline(midas_conv2d_spatial_int8_relu_spv,midas_conv2d_spatial_int8_relu_spv_size,6,20);
        reduce_absmax_.set_debug_name("zipdepth_reduce_absmax");
        quantize_int8_.set_debug_name("zipdepth_quantize_int8");
        spatial_int8_.set_debug_name("zipdepth_conv2d_spatial_int8");
        spatial_int8_relu_.set_debug_name("zipdepth_conv2d_spatial_int8_relu");
    }
}

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

void ZipDepthOps::spatial_int8(midas_native::VulkanBuffer&o,const midas_native::VulkanBuffer&i,const midas_native::VulkanBuffer&w,const midas_native::VulkanBuffer&ws,const midas_native::VulkanBuffer&b,std::uint32_t width,std::uint32_t height,std::uint32_t ic,std::uint32_t oc,bool has_bias,bool relu){
    if(!context_.supports_packed_int8_dot())throw std::runtime_error("packed INT8 dot product is not supported");
    if(ic%4!=0)throw std::invalid_argument("INT8 convolution channels must be divisible by four");
    auto& scale=int8_workspace_.scales(sizeof(float),
        [this](std::uint64_t bytes){return context_.create_device_buffer(bytes);});
    auto& packed=int8_workspace_.packed(
        std::uint64_t(width)*height*(ic/4)*sizeof(std::uint32_t),
        [this](std::uint64_t bytes){return context_.create_device_buffer(bytes);});
    const std::uint32_t count=width*height*ic;
    const std::uint32_t reduction_groups=std::min(256u,up(count,4096));
    struct R{std::uint32_t count;float divisor;};
    if(reduction_groups==1){
        const R reduce{count,127.0f};
        context_.dispatch(reduce_absmax_,{&i,&scale},&reduce,sizeof(reduce),1);
    }else{
        auto& partial=int8_workspace_.partial(
            std::uint64_t(reduction_groups)*sizeof(float),
            [this](std::uint64_t bytes){return context_.create_device_buffer(bytes);});
        const R first{count,1.0f};
        context_.dispatch(reduce_absmax_,{&i,&partial},&first,sizeof(first),reduction_groups);
        const R final{reduction_groups,127.0f};
        context_.dispatch(reduce_absmax_,{&partial,&scale},&final,sizeof(final),1);
    }
    struct Q{std::uint32_t w,h,c;}q{width,height,ic};
    context_.dispatch(quantize_int8_,{&packed,&i,&scale},&q,sizeof(q),up(width*height*(ic/4),256));
    struct P{std::uint32_t w,h,ic,oc,bias;}p{width,height,ic,oc,has_bias?1u:0u};
    context_.dispatch(relu?spatial_int8_relu_:spatial_int8_,{&o,&packed,&w,&scale,&ws,&b},&p,sizeof(p),up(width,8),up(height,8),oc);
}

void ZipDepthOps::pointwise_fp16_weights(midas_native::VulkanBuffer&o,const midas_native::VulkanBuffer&i,const midas_native::VulkanBuffer&w,const midas_native::VulkanBuffer&b,std::uint32_t spatial,std::uint32_t ic,std::uint32_t oc,bool has_bias){
    if(!context_.supports_float16())throw std::runtime_error("FP16 pointwise convolution is unavailable");
    struct P{std::uint32_t spatial,ic,oc,bias;}p{spatial,ic,oc,has_bias?1u:0u};
    context_.dispatch(pointwise_fp16_weights_,{&o,&i,&w,&b},&p,sizeof(p),up(spatial*oc,64));
}

}  // namespace zipdepth_native
