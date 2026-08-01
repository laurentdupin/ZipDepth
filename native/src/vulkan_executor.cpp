#include "vulkan_executor.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace zipdepth_native {
namespace {
std::uint64_t elements(std::uint32_t c,std::uint32_t h,std::uint32_t w){return std::uint64_t(c)*h*w;}
std::uint32_t checked(std::uint64_t n){if(n>std::numeric_limits<std::uint32_t>::max())throw std::overflow_error("ZipDepth tensor too large");return static_cast<std::uint32_t>(n);}
}

VulkanExecutor::VulkanExecutor(const std::string& path,std::uint32_t index)
    :model_(path),context_(index),operators_(context_),extra_(context_),
     zero_(context_.create_device_buffer(sizeof(float))){
    const float z=0;context_.upload(zero_,&z,sizeof(z));
    weights_.reserve(model_.tensor_names().size());
    for(const std::string&name:model_.tensor_names()){
        const auto&t=model_.tensor(name);auto b=context_.create_device_buffer(t.elements*sizeof(float));
        context_.upload(b,t.data,static_cast<std::size_t>(t.elements*sizeof(float)));
        weights_.emplace(name,std::move(b));
    }
}
const midas_native::VulkanBuffer& VulkanExecutor::weight(const std::string&n)const{auto i=weights_.find(n);if(i==weights_.end())throw std::runtime_error("missing GPU tensor: "+n);return i->second;}
VulkanExecutor::Tensor VulkanExecutor::make(std::uint32_t c,std::uint32_t h,std::uint32_t w){return {c,h,w,context_.create_device_buffer(elements(c,h,w)*sizeof(float))};}
VulkanExecutor::Tensor VulkanExecutor::conv(const Tensor&i,const std::string&n,const char*bias,std::uint32_t stride,std::uint32_t padding,std::uint32_t dilation,std::uint32_t groups){const auto&s=model_.tensor(n);if(s.rank!=4||s.dimensions[1]!=i.channels/groups)throw std::runtime_error("GPU convolution shape mismatch: "+n);auto oc=static_cast<std::uint32_t>(s.dimensions[0]),kh=static_cast<std::uint32_t>(s.dimensions[2]),kw=static_cast<std::uint32_t>(s.dimensions[3]);auto oh=(i.height+2*padding-dilation*(kh-1)-1)/stride+1,ow=(i.width+2*padding-dilation*(kw-1)-1)/stride+1;Tensor o=make(oc,oh,ow);operators_.conv(o.buffer,i.buffer,weight(n),bias?weight(bias):zero_,i.width,i.height,i.channels,ow,oh,oc,kh,kw,stride,padding,padding,dilation,groups,bias!=nullptr);return o;}
VulkanExecutor::Tensor VulkanExecutor::bn(const Tensor&i,const std::string&p,bool relu){Tensor o=make(i.channels,i.height,i.width);operators_.batch_norm_activation(o.buffer,i.buffer,weight(p+".weight"),weight(p+".bias"),weight(p+".running_mean"),weight(p+".running_var"),checked(elements(i.channels,i.height,i.width)),i.width*i.height,relu?1u:0u);return o;}
VulkanExecutor::Tensor VulkanExecutor::conv_bn(const Tensor&i,const std::string&p,std::uint32_t stride,bool relu){const auto&s=model_.tensor(p+".conv.weight");return bn(conv(i,p+".conv.weight",nullptr,stride,static_cast<std::uint32_t>(s.dimensions[2]/2)),p+".bn",relu);}
VulkanExecutor::Tensor VulkanExecutor::add(const Tensor&a,const Tensor&b,float scale){if(a.channels!=b.channels||a.height!=b.height||a.width!=b.width)throw std::runtime_error("GPU add shape mismatch");Tensor o=make(a.channels,a.height,a.width);if(scale==1.0f)operators_.add(o.buffer,a.buffer,b.buffer,checked(elements(a.channels,a.height,a.width)));else extra_.elementwise(o.buffer,a.buffer,b.buffer,checked(elements(a.channels,a.height,a.width)),a.width*a.height,1,scale);return o;}
VulkanExecutor::Tensor VulkanExecutor::rep(const Tensor&i,const std::string&p,std::uint32_t stride){Tensor a=bn(conv(i,p+".branch_3x3.0.weight",nullptr,stride,1),p+".branch_3x3.1",false);Tensor b=bn(conv(i,p+".branch_1x1.0.weight",nullptr,stride,0),p+".branch_1x1.1",false);Tensor o=add(a,b);if(stride==1&&i.channels==o.channels)o=add(o,i);Tensor r=make(o.channels,o.height,o.width);operators_.activation(r.buffer,o.buffer,checked(elements(o.channels,o.height,o.width)),1);return r;}
VulkanExecutor::Tensor VulkanExecutor::resize(const Tensor&i,std::uint32_t w,std::uint32_t h,bool ac){Tensor o=make(i.channels,h,w);operators_.resize(o.buffer,i.buffer,i.width,i.height,w,h,i.channels,ac);return o;}
VulkanExecutor::Tensor VulkanExecutor::nearest(const Tensor&i,std::uint32_t w,std::uint32_t h){Tensor o=make(i.channels,h,w);extra_.nearest(o.buffer,i.buffer,i.width,i.height,w,h,i.channels);return o;}
VulkanExecutor::Tensor VulkanExecutor::adaptive(const Tensor&i,std::uint32_t w,std::uint32_t h){Tensor o=make(i.channels,h,w);extra_.adaptive_average(o.buffer,i.buffer,i.width,i.height,w,h,i.channels);return o;}
VulkanExecutor::Tensor VulkanExecutor::fusion(const Tensor&high,const Tensor&low,const std::string&p){auto oc=static_cast<std::uint32_t>(model_.tensor(p+".proj_high.weight").dimensions[0]);std::uint32_t gh=high.channels%4==0&&oc%4==0?4:1,gl=low.channels%4==0&&oc%4==0?4:1;Tensor h=conv(high,p+".proj_high.weight",nullptr,1,0,1,gh),l=conv(low,p+".proj_low.weight",nullptr,1,0,1,gl);l=resize(l,high.width,high.height);return bn(add(h,l),p+".bn",true);}

VulkanExecutor::Tensor VulkanExecutor::infer_device(midas_native::VulkanBuffer input,std::uint32_t width,std::uint32_t height){
    if(!width||!height||width%32||height%32)throw std::invalid_argument("ZipDepth GPU input must be multiple of 32");
    Tensor output;
    context_.batch([&]{
        Tensor x{3,height,width,std::move(input)};
        Tensor half=conv_bn(x,"encoder.stem_half",2),q=conv_bn(half,"encoder.stem_quarter",2);
        Tensor s1=rep(rep(q,"encoder.stage1.0"),"encoder.stage1.1");
        Tensor s2=rep(s1,"encoder.down2",2);s2=rep(rep(s2,"encoder.stage2.0"),"encoder.stage2.1");
        Tensor m1=conv(s2,"encoder.stage2.2.branch1.weight",nullptr,1,1,1,s2.channels),m2=conv(s2,"encoder.stage2.2.branch2.weight",nullptr,1,2,2,s2.channels);
        s2=add(s2,bn(add(m1,m2),"encoder.stage2.2.bn",false));
        Tensor stripped=make(s2.channels,s2.height,s2.width);extra_.strip_attention(stripped.buffer,s2.buffer,weight("encoder.stage2.3.gate_conv.0.weight"),weight("encoder.stage2.3.gate_conv.1.weight"),weight("encoder.stage2.3.gate_conv.1.bias"),weight("encoder.stage2.3.gate_conv.1.running_mean"),weight("encoder.stage2.3.gate_conv.1.running_var"),s2.width,s2.height,s2.channels);s2=std::move(stripped);
        Tensor s3=rep(s2,"encoder.down3",2);for(std::uint32_t i=0;i<6;++i)s3=rep(s3,"encoder.stage3."+std::to_string(i));
        Tensor avg=make(s3.channels,1,1);extra_.channel_average(avg.buffer,s3.buffer,s3.channels,s3.width*s3.height);
        Tensor ca=conv(avg,"encoder.stage3.6.fc.0.weight",nullptr);Tensor car=make(ca.channels,1,1);operators_.activation(car.buffer,ca.buffer,ca.channels,1);ca=conv(car,"encoder.stage3.6.fc.2.weight",nullptr);Tensor gate=make(ca.channels,1,1);extra_.elementwise(gate.buffer,ca.buffer,ca.buffer,ca.channels,1,4);Tensor weighted=make(s3.channels,s3.height,s3.width);extra_.elementwise(weighted.buffer,s3.buffer,gate.buffer,checked(elements(s3.channels,s3.height,s3.width)),s3.width*s3.height,2);s3=std::move(weighted);
        Tensor logits=conv(s3,"encoder.stage3.7.context_weight.weight","encoder.stage3.7.context_weight.bias");Tensor ctx=make(s3.channels,1,1);extra_.context_reduce(ctx.buffer,s3.buffer,logits.buffer,s3.channels,s3.width*s3.height);ctx=conv(ctx,"encoder.stage3.7.transform.0.weight","encoder.stage3.7.transform.0.bias");ctx=bn(ctx,"encoder.stage3.7.transform.1",true);ctx=conv(ctx,"encoder.stage3.7.transform.3.weight","encoder.stage3.7.transform.3.bias");Tensor broadcast=make(s3.channels,s3.height,s3.width);extra_.elementwise(broadcast.buffer,s3.buffer,ctx.buffer,checked(elements(s3.channels,s3.height,s3.width)),s3.width*s3.height,3);s3=std::move(broadcast);
        Tensor s4=rep(s3,"encoder.down4",2);s4=rep(rep(s4,"encoder.stage4.0"),"encoder.stage4.1");Tensor spp=conv_bn(s4,"encoder.spp.cv1"),p1=make(spp.channels,spp.height,spp.width),p2=make(spp.channels,spp.height,spp.width),p3=make(spp.channels,spp.height,spp.width);extra_.maxpool5(p1.buffer,spp.buffer,spp.width,spp.height,spp.channels);extra_.maxpool5(p2.buffer,p1.buffer,p1.width,p1.height,p1.channels);extra_.maxpool5(p3.buffer,p2.buffer,p2.width,p2.height,p2.channels);Tensor joined=make(spp.channels*4,spp.height,spp.width);extra_.concat4(joined.buffer,spp.buffer,p1.buffer,p2.buffer,p3.buffer,checked(elements(spp.channels,spp.height,spp.width)));s4=conv_bn(joined,"encoder.spp.cv2");
        Tensor lth=conv(s4,"encoder.cross_scale.low_to_high.weight",nullptr,1,0,1,4);lth=nearest(lth,s3.width,s3.height);Tensor htl=conv(s3,"encoder.cross_scale.high_to_low.weight",nullptr,1,0,1,4);htl=adaptive(htl,s4.width,s4.height);s3=add(s3,lth,.3f);s4=add(s4,htl,.3f);
        Tensor f4=conv_bn(s4,"decoder.proj4"),f3=fusion(s3,f4,"decoder.fuse3"),f2=fusion(s2,f3,"decoder.fuse2"),f1=fusion(s1,f2,"decoder.fuse1"),fh=fusion(half,f1,"decoder.fuse_half"),depth=conv(fh,"decoder.head_half.weight","decoder.head_half.bias",1,1);
        output=make(1,height,width);
        if(model_.kind()==ZIPDEPTH_MODEL_BASE_GPU){Tensor hidden=conv(fh,"decoder.convex_up.mask_pred.0.weight",nullptr,1,1);hidden=bn(hidden,"decoder.convex_up.mask_pred.1",true);Tensor weights=conv(hidden,"decoder.convex_up.mask_pred.3.weight","decoder.convex_up.mask_pred.3.bias");extra_.convex(output.buffer,depth.buffer,weights.buffer,depth.width,depth.height);}else{Tensor alpha=conv(fh,"decoder.convex_up.where_conv.0.weight",nullptr);alpha=bn(alpha,"decoder.convex_up.where_conv.1",true);alpha=conv(alpha,"decoder.convex_up.where_conv.3.weight",nullptr,1,2,1,alpha.channels);alpha=bn(alpha,"decoder.convex_up.where_conv.4",true);alpha=conv(alpha,"decoder.convex_up.where_conv.6.weight",nullptr);extra_.mobile(output.buffer,depth.buffer,alpha.buffer,depth.width,depth.height);}
    });
    return output;
}

}  // namespace zipdepth_native
