#include "cpu_executor.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace zipdepth_native {
namespace {

std::uint64_t count(std::uint32_t c, std::uint32_t h, std::uint32_t w) {
    return std::uint64_t(c) * h * w;
}

CpuTensor add(const CpuTensor& a, const CpuTensor& b, float b_scale = 1.0f) {
    if (a.channels != b.channels || a.height != b.height || a.width != b.width)
        throw std::runtime_error("ZipDepth add shape mismatch");
    CpuTensor out{a.channels, a.height, a.width, a.data};
    for (std::size_t i = 0; i < out.data.size(); ++i)
        out.data[i] += b.data[i] * b_scale;
    return out;
}

void relu(CpuTensor& tensor) {
    for (float& value : tensor.data) value = std::max(value, 0.0f);
}

void sigmoid(CpuTensor& tensor) {
    for (float& value : tensor.data) value = 1.0f / (1.0f + std::exp(-value));
}

CpuTensor resize_nearest(const CpuTensor& input, std::uint32_t width,
                         std::uint32_t height) {
    CpuTensor out{input.channels, height, width,
                  std::vector<float>(count(input.channels, height, width))};
    for (std::uint32_t c = 0; c < input.channels; ++c)
        for (std::uint32_t y = 0; y < height; ++y) {
            const std::uint32_t sy = std::min(
                input.height - 1,
                static_cast<std::uint32_t>(std::uint64_t(y) * input.height / height));
            for (std::uint32_t x = 0; x < width; ++x) {
                const std::uint32_t sx = std::min(
                    input.width - 1,
                    static_cast<std::uint32_t>(std::uint64_t(x) * input.width / width));
                out.data[(std::uint64_t(c) * height + y) * width + x] =
                    input.data[(std::uint64_t(c) * input.height + sy) * input.width + sx];
            }
        }
    return out;
}

CpuTensor adaptive_avg(const CpuTensor& input, std::uint32_t width,
                       std::uint32_t height) {
    CpuTensor out{input.channels, height, width,
                  std::vector<float>(count(input.channels, height, width))};
    for (std::uint32_t c = 0; c < input.channels; ++c)
        for (std::uint32_t y = 0; y < height; ++y) {
            const std::uint32_t y0 = y * input.height / height;
            const std::uint32_t y1 = ((y + 1) * input.height + height - 1) / height;
            for (std::uint32_t x = 0; x < width; ++x) {
                const std::uint32_t x0 = x * input.width / width;
                const std::uint32_t x1 = ((x + 1) * input.width + width - 1) / width;
                float sum = 0.0f;
                for (std::uint32_t sy = y0; sy < y1; ++sy)
                    for (std::uint32_t sx = x0; sx < x1; ++sx)
                        sum += input.data[(std::uint64_t(c) * input.height + sy) *
                                          input.width + sx];
                out.data[(std::uint64_t(c) * height + y) * width + x] =
                    sum / static_cast<float>((y1 - y0) * (x1 - x0));
            }
        }
    return out;
}

CpuTensor concat(const std::vector<const CpuTensor*>& inputs) {
    if (inputs.empty()) throw std::invalid_argument("empty ZipDepth concat");
    const auto h = inputs[0]->height, w = inputs[0]->width;
    std::uint32_t channels = 0;
    for (const CpuTensor* input : inputs) {
        if (input->height != h || input->width != w)
            throw std::runtime_error("ZipDepth concat shape mismatch");
        channels += input->channels;
    }
    CpuTensor out{channels, h, w, std::vector<float>(count(channels, h, w))};
    std::size_t offset = 0;
    for (const CpuTensor* input : inputs) {
        std::copy(input->data.begin(), input->data.end(), out.data.begin() + offset);
        offset += input->data.size();
    }
    return out;
}

CpuTensor max_pool5(const CpuTensor& input) {
    CpuTensor out{input.channels, input.height, input.width,
                  std::vector<float>(input.data.size())};
    for (std::uint32_t c = 0; c < input.channels; ++c)
        for (std::uint32_t y = 0; y < input.height; ++y)
            for (std::uint32_t x = 0; x < input.width; ++x) {
                float value = -std::numeric_limits<float>::infinity();
                for (int ky = -2; ky <= 2; ++ky)
                    for (int kx = -2; kx <= 2; ++kx) {
                        const auto sy = static_cast<std::uint32_t>(std::clamp<int>(
                            static_cast<int>(y) + ky, 0, input.height - 1));
                        const auto sx = static_cast<std::uint32_t>(std::clamp<int>(
                            static_cast<int>(x) + kx, 0, input.width - 1));
                        value = std::max(value, input.data[
                            (std::uint64_t(c) * input.height + sy) * input.width + sx]);
                    }
                out.data[(std::uint64_t(c) * input.height + y) * input.width + x] = value;
            }
    return out;
}

}  // namespace

CpuTensor resize_bilinear(const CpuTensor& input, std::uint32_t width,
                          std::uint32_t height, bool align_corners) {
    CpuTensor out{input.channels, height, width,
                  std::vector<float>(count(input.channels, height, width))};
    for (std::uint32_t y = 0; y < height; ++y) {
        float fy = align_corners && height > 1
            ? float(y) * (input.height - 1) / float(height - 1)
            : (float(y) + 0.5f) * input.height / float(height) - 0.5f;
        fy = std::clamp(fy, 0.0f, float(input.height - 1));
        const auto y0 = static_cast<std::uint32_t>(fy);
        const auto y1 = std::min(y0 + 1, input.height - 1);
        const float wy = fy - y0;
        for (std::uint32_t x = 0; x < width; ++x) {
            float fx = align_corners && width > 1
                ? float(x) * (input.width - 1) / float(width - 1)
                : (float(x) + 0.5f) * input.width / float(width) - 0.5f;
            fx = std::clamp(fx, 0.0f, float(input.width - 1));
            const auto x0 = static_cast<std::uint32_t>(fx);
            const auto x1 = std::min(x0 + 1, input.width - 1);
            const float wx = fx - x0;
            for (std::uint32_t c = 0; c < input.channels; ++c) {
                const auto base = std::uint64_t(c) * input.height * input.width;
                const float top = input.data[base + y0 * input.width + x0] * (1-wx) +
                                  input.data[base + y0 * input.width + x1] * wx;
                const float bottom = input.data[base + y1 * input.width + x0] * (1-wx) +
                                     input.data[base + y1 * input.width + x1] * wx;
                out.data[(std::uint64_t(c) * height + y) * width + x] =
                    top * (1-wy) + bottom * wy;
            }
        }
    }
    return out;
}

CpuTensor CpuExecutor::conv(const CpuTensor& input, const std::string& name,
                            const char* bias_name, std::uint32_t stride,
                            std::uint32_t padding, std::uint32_t dilation,
                            std::uint32_t groups) const {
    const TensorView& weight = model_.tensor(name);
    if (weight.rank != 4 || groups == 0 || input.channels % groups != 0 ||
        weight.dimensions[1] != input.channels / groups)
        throw std::runtime_error("ZipDepth convolution shape mismatch: " + name);
    const auto out_c = static_cast<std::uint32_t>(weight.dimensions[0]);
    const auto kh = static_cast<std::uint32_t>(weight.dimensions[2]);
    const auto kw = static_cast<std::uint32_t>(weight.dimensions[3]);
    const auto out_h = (input.height + 2 * padding - dilation * (kh - 1) - 1) /
        stride + 1;
    const auto out_w = (input.width + 2 * padding - dilation * (kw - 1) - 1) /
        stride + 1;
    CpuTensor out{out_c, out_h, out_w,
                  std::vector<float>(count(out_c, out_h, out_w))};
    const TensorView* bias = bias_name ? &model_.tensor(bias_name) : nullptr;
    const auto in_per_group = input.channels / groups;
    const auto out_per_group = out_c / groups;
    for (std::uint32_t oc = 0; oc < out_c; ++oc) {
        const auto group = oc / out_per_group;
        for (std::uint32_t oy = 0; oy < out_h; ++oy)
            for (std::uint32_t ox = 0; ox < out_w; ++ox) {
                float sum = bias ? bias->data[oc] : 0.0f;
                for (std::uint32_t icg = 0; icg < in_per_group; ++icg)
                    for (std::uint32_t ky = 0; ky < kh; ++ky)
                        for (std::uint32_t kx = 0; kx < kw; ++kx) {
                            const int iy = int(oy * stride + ky * dilation) - int(padding);
                            const int ix = int(ox * stride + kx * dilation) - int(padding);
                            if (iy < 0 || ix < 0 || iy >= int(input.height) || ix >= int(input.width))
                                continue;
                            const auto ic = group * in_per_group + icg;
                            sum += input.data[(std::uint64_t(ic) * input.height + iy) * input.width + ix] *
                                weight.data[((std::uint64_t(oc) * in_per_group + icg) * kh + ky) * kw + kx];
                        }
                out.data[(std::uint64_t(oc) * out_h + oy) * out_w + ox] = sum;
            }
    }
    return out;
}

CpuTensor CpuExecutor::bn(const CpuTensor& input, const std::string& prefix,
                          bool apply_relu) const {
    const auto& gamma = model_.tensor(prefix + ".weight");
    const auto& beta = model_.tensor(prefix + ".bias");
    const auto& mean = model_.tensor(prefix + ".running_mean");
    const auto& variance = model_.tensor(prefix + ".running_var");
    CpuTensor out{input.channels, input.height, input.width, input.data};
    const auto plane = std::uint64_t(input.height) * input.width;
    for (std::uint32_t c = 0; c < input.channels; ++c) {
        const float scale = gamma.data[c] / std::sqrt(variance.data[c] + 1.0e-5f);
        const float shift = beta.data[c] - mean.data[c] * scale;
        for (std::uint64_t i = 0; i < plane; ++i) {
            float& value = out.data[std::uint64_t(c) * plane + i];
            value = value * scale + shift;
            if (apply_relu) value = std::max(value, 0.0f);
        }
    }
    return out;
}

CpuTensor CpuExecutor::conv_bn(const CpuTensor& input, const std::string& prefix,
                               std::uint32_t stride, bool apply_relu) const {
    const auto& weight = model_.tensor(prefix + ".conv.weight");
    return bn(conv(input, prefix + ".conv.weight", nullptr, stride,
                   static_cast<std::uint32_t>(weight.dimensions[2] / 2)),
              prefix + ".bn", apply_relu);
}

CpuTensor CpuExecutor::rep(const CpuTensor& input, const std::string& prefix,
                           std::uint32_t stride) const {
    CpuTensor a = bn(conv(input, prefix + ".branch_3x3.0.weight", nullptr,
                          stride, 1), prefix + ".branch_3x3.1", false);
    CpuTensor b = bn(conv(input, prefix + ".branch_1x1.0.weight", nullptr,
                          stride, 0), prefix + ".branch_1x1.1", false);
    CpuTensor out = add(a, b);
    if (stride == 1 && input.channels == out.channels) out = add(out, input);
    relu(out);
    return out;
}

CpuTensor CpuExecutor::fusion(const CpuTensor& high, const CpuTensor& low,
                              const std::string& prefix) const {
    const auto high_groups = static_cast<std::uint32_t>(
        high.channels % 4 == 0 && model_.tensor(prefix + ".proj_high.weight").dimensions[0] % 4 == 0 ? 4 : 1);
    const auto low_groups = static_cast<std::uint32_t>(
        low.channels % 4 == 0 && model_.tensor(prefix + ".proj_low.weight").dimensions[0] % 4 == 0 ? 4 : 1);
    CpuTensor h = conv(high, prefix + ".proj_high.weight", nullptr, 1, 0, 1, high_groups);
    CpuTensor l = conv(low, prefix + ".proj_low.weight", nullptr, 1, 0, 1, low_groups);
    l = resize_bilinear(l, high.width, high.height, false);
    return bn(add(h, l), prefix + ".bn", true);
}

CpuTensor CpuExecutor::infer(const float* rgb, std::uint32_t width,
                             std::uint32_t height) const {
    if (!rgb || !width || !height || width % 32 || height % 32)
        throw std::invalid_argument("ZipDepth input must be a positive multiple of 32");
    CpuTensor x{3, height, width, std::vector<float>(rgb, rgb + count(3,height,width))};
    const auto& mean = model_.tensor("mean");
    const auto& stddev = model_.tensor("std");
    for (std::uint32_t c = 0; c < 3; ++c)
        for (std::uint64_t i = 0; i < std::uint64_t(width) * height; ++i)
            x.data[std::uint64_t(c) * width * height + i] =
                (x.data[std::uint64_t(c) * width * height + i] - mean.data[c]) /
                stddev.data[c];
    CpuTensor half = conv_bn(x, "encoder.stem_half", 2);
    CpuTensor q = conv_bn(half, "encoder.stem_quarter", 2);
    CpuTensor s1 = rep(rep(q, "encoder.stage1.0"), "encoder.stage1.1");
    CpuTensor s2 = rep(s1, "encoder.down2", 2);
    s2 = rep(rep(s2, "encoder.stage2.0"), "encoder.stage2.1");
    CpuTensor multi1 = conv(s2, "encoder.stage2.2.branch1.weight", nullptr, 1, 1, 1, s2.channels);
    CpuTensor multi2 = conv(s2, "encoder.stage2.2.branch2.weight", nullptr, 1, 2, 2, s2.channels);
    s2 = add(s2, bn(add(multi1, multi2), "encoder.stage2.2.bn", false));
    CpuTensor strips{ s2.channels, s2.height, s2.width,
                      std::vector<float>(s2.data.size()) };
    for (std::uint32_t c=0;c<s2.channels;++c)
        for(std::uint32_t y=0;y<s2.height;++y)
            for(std::uint32_t x0=0;x0<s2.width;++x0){
                float hmean=0,wmean=0;
                for(std::uint32_t x1=0;x1<s2.width;++x1)hmean+=s2.data[(std::uint64_t(c)*s2.height+y)*s2.width+x1];
                for(std::uint32_t y1=0;y1<s2.height;++y1)wmean+=s2.data[(std::uint64_t(c)*s2.height+y1)*s2.width+x0];
                strips.data[(std::uint64_t(c)*s2.height+y)*s2.width+x0]=hmean/s2.width+wmean/s2.height;
            }
    CpuTensor gate = bn(conv(strips,"encoder.stage2.3.gate_conv.0.weight",nullptr,1,0,1,s2.channels),"encoder.stage2.3.gate_conv.1",false);
    sigmoid(gate);
    for(std::size_t i=0;i<s2.data.size();++i)s2.data[i]*=gate.data[i];
    CpuTensor s3 = rep(s2, "encoder.down3", 2);
    for(std::uint32_t i=0;i<6;++i)s3=rep(s3,"encoder.stage3."+std::to_string(i));
    CpuTensor pooled=adaptive_avg(s3,1,1);
    CpuTensor ca=conv(pooled,"encoder.stage3.6.fc.0.weight",nullptr);
    relu(ca);ca=conv(ca,"encoder.stage3.6.fc.2.weight",nullptr);sigmoid(ca);
    for(std::uint32_t c=0;c<s3.channels;++c)for(std::uint64_t i=0;i<std::uint64_t(s3.width)*s3.height;++i)s3.data[std::uint64_t(c)*s3.width*s3.height+i]*=ca.data[c];
    CpuTensor mask=conv(s3,"encoder.stage3.7.context_weight.weight","encoder.stage3.7.context_weight.bias");
    float maxv=*std::max_element(mask.data.begin(),mask.data.end()),den=0;
    for(float&v:mask.data){v=std::exp(v-maxv);den+=v;}for(float&v:mask.data)v/=den;
    CpuTensor context{s3.channels,1,1,std::vector<float>(s3.channels)};
    for(std::uint32_t c=0;c<s3.channels;++c)for(std::uint64_t i=0;i<mask.data.size();++i)context.data[c]+=s3.data[std::uint64_t(c)*mask.data.size()+i]*mask.data[i];
    context=conv(context,"encoder.stage3.7.transform.0.weight","encoder.stage3.7.transform.0.bias");
    context=bn(context,"encoder.stage3.7.transform.1",true);
    context=conv(context,"encoder.stage3.7.transform.3.weight","encoder.stage3.7.transform.3.bias");
    for(std::uint32_t c=0;c<s3.channels;++c)for(std::uint64_t i=0;i<std::uint64_t(s3.width)*s3.height;++i)s3.data[std::uint64_t(c)*s3.width*s3.height+i]+=context.data[c];
    CpuTensor s4=rep(s3,"encoder.down4",2);
    s4=rep(rep(s4,"encoder.stage4.0"),"encoder.stage4.1");
    CpuTensor spp=conv_bn(s4,"encoder.spp.cv1");
    CpuTensor p1=max_pool5(spp),p2=max_pool5(p1),p3=max_pool5(p2);
    s4=conv_bn(concat({&spp,&p1,&p2,&p3}),"encoder.spp.cv2");
    CpuTensor low_to_high=conv(s4,"encoder.cross_scale.low_to_high.weight",nullptr,1,0,1,4);
    low_to_high=resize_nearest(low_to_high,s3.width,s3.height);
    CpuTensor high_to_low=conv(s3,"encoder.cross_scale.high_to_low.weight",nullptr,1,0,1,4);
    high_to_low=adaptive_avg(high_to_low,s4.width,s4.height);
    s3=add(s3,low_to_high,0.3f);s4=add(s4,high_to_low,0.3f);
    CpuTensor f4=conv_bn(s4,"decoder.proj4");
    CpuTensor f3=fusion(s3,f4,"decoder.fuse3");
    CpuTensor f2=fusion(s2,f3,"decoder.fuse2");
    CpuTensor f1=fusion(s1,f2,"decoder.fuse1");
    CpuTensor fhalf=fusion(half,f1,"decoder.fuse_half");
    CpuTensor depth=conv(fhalf,"decoder.head_half.weight","decoder.head_half.bias",1,1);
    CpuTensor output{1,height,width,std::vector<float>(std::uint64_t(width)*height)};
    if(model_.kind()==ZIPDEPTH_MODEL_BASE_GPU){
        CpuTensor hidden=conv(fhalf,"decoder.convex_up.mask_pred.0.weight",nullptr,1,1);
        hidden=bn(hidden,"decoder.convex_up.mask_pred.1",true);
        CpuTensor weights=conv(hidden,"decoder.convex_up.mask_pred.3.weight","decoder.convex_up.mask_pred.3.bias");
        for(std::uint32_t y=0;y<height;++y)for(std::uint32_t x0=0;x0<width;++x0){const std::uint32_t py=y/2,px=x0/2,s=(y%2)*2+x0%2;float mx=-std::numeric_limits<float>::infinity();for(std::uint32_t n=0;n<9;++n)mx=std::max(mx,weights.data[(std::uint64_t(n*4+s)*depth.height+py)*depth.width+px]);float total=0,value=0;for(std::uint32_t n=0;n<9;++n){float w=std::exp(weights.data[(std::uint64_t(n*4+s)*depth.height+py)*depth.width+px]-mx);total+=w;int ny=std::clamp<int>(int(py)+int(n/3)-1,0,depth.height-1),nx=std::clamp<int>(int(px)+int(n%3)-1,0,depth.width-1);value+=w*depth.data[std::uint64_t(ny)*depth.width+nx];}output.data[std::uint64_t(y)*width+x0]=std::max(value/total,0.0f);}
    }else{
        CpuTensor alpha=conv(fhalf,"decoder.convex_up.where_conv.0.weight",nullptr);
        alpha=bn(alpha,"decoder.convex_up.where_conv.1",true);
        alpha=conv(alpha,"decoder.convex_up.where_conv.3.weight",nullptr,1,2,1,alpha.channels);
        alpha=bn(alpha,"decoder.convex_up.where_conv.4",true);
        alpha=conv(alpha,"decoder.convex_up.where_conv.6.weight",nullptr);alpha=resize_bilinear(alpha,width,height,false);sigmoid(alpha);
        CpuTensor nn=resize_nearest(depth,width,height),bi=resize_bilinear(depth,width,height,false);
        for(std::size_t i=0;i<output.data.size();++i)output.data[i]=std::max(alpha.data[i]*nn.data[i]+(1-alpha.data[i])*bi.data[i],0.0f);
    }
    return output;
}

}  // namespace zipdepth_native
