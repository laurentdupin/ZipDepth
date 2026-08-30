#include "metal_executor.h"
#include "model.h"

#include <inferbridge/native_harness_precision.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace zipdepth_native {
namespace {

MPSShape* shape(std::initializer_list<NSInteger> values) {
    NSMutableArray<NSNumber*>* result =
        [NSMutableArray arrayWithCapacity:values.size()];
    for (NSInteger value : values) [result addObject:@(value)];
    return result;
}
MPSShape* shape(const TensorView& tensor) {
    NSMutableArray<NSNumber*>* result =
        [NSMutableArray arrayWithCapacity:tensor.rank];
    for (std::uint32_t i = 0; i < tensor.rank; ++i)
        [result addObject:@(tensor.dimensions[i])];
    return result;
}

struct Tensor {
    MPSGraphTensor* value = nil;
    int channels = 0;
    int height = 0;
    int width = 0;
};

class GraphBuilder {
public:
    GraphBuilder(const ModelFile& model, int width, int height, bool fp16)
        : model_(model), width_(width), height_(height), fp16_(fp16),
          graph_([MPSGraph new]) {}
    MPSGraph* graph() const { return graph_; }
    MPSGraphTensor* input() const { return input_; }
    NSArray<MPSGraphTensor*>* outputs() const { return @[depth_, auxiliary_]; }
    zipdepth_model_kind kind() const { return model_.kind(); }

    void build() {
        input_ = [graph_ placeholderWithShape:shape({1, 3, height_, width_})
            dataType:MPSDataTypeFloat32 name:@"rgb_chw"];
        MPSGraphTensor* value = fp16_ ? [graph_ castTensor:input_
            toType:MPSDataTypeFloat16 name:nil] : input_;
        value = [graph_ subtractionWithPrimaryTensor:value
            secondaryTensor:channel_constant("mean") name:nil];
        value = [graph_ divisionWithPrimaryTensor:value
            secondaryTensor:channel_constant("std") name:nil];
        Tensor x{value, 3, height_, width_};
        Tensor half = conv_bn(x, "encoder.stem_half", 2);
        Tensor quarter = conv_bn(half, "encoder.stem_quarter", 2);
        Tensor s1 = rep(rep(quarter, "encoder.stage1.0"), "encoder.stage1.1");
        Tensor s2 = rep(s1, "encoder.down2", 2);
        s2 = rep(rep(s2, "encoder.stage2.0"), "encoder.stage2.1");
        Tensor branch1 = conv(s2, "encoder.stage2.2.branch1.weight", nullptr,
            1, 1, 1, s2.channels);
        Tensor branch2 = conv(s2, "encoder.stage2.2.branch2.weight", nullptr,
            1, 2, 2, s2.channels);
        s2 = add(s2, batch_norm(add(branch1, branch2),
            "encoder.stage2.2.bn", false));
        MPSGraphTensor* horizontal = [graph_ meanOfTensor:s2.value axes:@[@3] name:nil];
        MPSGraphTensor* vertical = [graph_ meanOfTensor:s2.value axes:@[@2] name:nil];
        Tensor strips{add_value(horizontal, vertical), s2.channels, s2.height, s2.width};
        Tensor gate = conv(strips, "encoder.stage2.3.gate_conv.0.weight",
            nullptr, 1, 0, 1, s2.channels);
        gate = batch_norm(gate, "encoder.stage2.3.gate_conv.1", false);
        gate.value = [graph_ sigmoidWithTensor:gate.value name:nil];
        s2.value = multiply_value(s2.value, gate.value);

        Tensor s3 = rep(s2, "encoder.down3", 2);
        for (int i = 0; i < 6; ++i)
            s3 = rep(s3, "encoder.stage3." + std::to_string(i));
        Tensor pooled{[graph_ meanOfTensor:s3.value axes:@[@2, @3] name:nil],
            s3.channels, 1, 1};
        Tensor ca = conv(pooled, "encoder.stage3.6.fc.0.weight", nullptr);
        ca.value = [graph_ reLUWithTensor:ca.value name:nil];
        ca = conv(ca, "encoder.stage3.6.fc.2.weight", nullptr);
        ca.value = [graph_ sigmoidWithTensor:ca.value name:nil];
        s3.value = multiply_value(s3.value, ca.value);
        Tensor logits = conv(s3, "encoder.stage3.7.context_weight.weight",
            "encoder.stage3.7.context_weight.bias");
        MPSGraphTensor* mask = [graph_ reshapeTensor:logits.value
            withShape:shape({1, s3.height * s3.width}) name:nil];
        mask = [graph_ softMaxWithTensor:mask axis:-1 name:nil];
        mask = [graph_ reshapeTensor:mask
            withShape:shape({1, 1, s3.height, s3.width}) name:nil];
        Tensor context{[graph_ reductionSumWithTensor:
            multiply_value(s3.value, mask) axes:@[@2, @3] name:nil],
            s3.channels, 1, 1};
        context = conv(context, "encoder.stage3.7.transform.0.weight",
            "encoder.stage3.7.transform.0.bias");
        context = batch_norm(context, "encoder.stage3.7.transform.1", true);
        context = conv(context, "encoder.stage3.7.transform.3.weight",
            "encoder.stage3.7.transform.3.bias");
        s3.value = add_value(s3.value, context.value);

        Tensor s4 = rep(s3, "encoder.down4", 2);
        s4 = rep(rep(s4, "encoder.stage4.0"), "encoder.stage4.1");
        Tensor spp = conv_bn(s4, "encoder.spp.cv1");
        Tensor p1 = max_pool5(spp), p2 = max_pool5(p1), p3 = max_pool5(p2);
        Tensor joined{[graph_ concatTensors:@[spp.value, p1.value, p2.value, p3.value]
            dimension:1 name:nil], spp.channels * 4, spp.height, spp.width};
        s4 = conv_bn(joined, "encoder.spp.cv2");
        Tensor low_to_high = conv(s4, "encoder.cross_scale.low_to_high.weight",
            nullptr, 1, 0, 1, 4);
        low_to_high = resize(low_to_high, s3.height, s3.width, MPSGraphResizeNearest, false);
        Tensor high_to_low = conv(s3, "encoder.cross_scale.high_to_low.weight",
            nullptr, 1, 0, 1, 4);
        high_to_low = average_pool2(high_to_low);
        s3 = add(s3, low_to_high, 0.3f);
        s4 = add(s4, high_to_low, 0.3f);

        Tensor f4 = conv_bn(s4, "decoder.proj4");
        Tensor f3 = fusion(s3, f4, "decoder.fuse3");
        Tensor f2 = fusion(s2, f3, "decoder.fuse2");
        Tensor f1 = fusion(s1, f2, "decoder.fuse1");
        Tensor fhalf = fusion(half, f1, "decoder.fuse_half");
        Tensor depth = conv(fhalf, "decoder.head_half.weight",
            "decoder.head_half.bias", 1, 1);
        depth_ = depth.value;
        if (model_.kind() == ZIPDEPTH_MODEL_BASE_GPU) {
            Tensor hidden = conv(fhalf, "decoder.convex_up.mask_pred.0.weight",
                nullptr, 1, 1);
            hidden = batch_norm(hidden, "decoder.convex_up.mask_pred.1", true);
            Tensor weights = conv(hidden, "decoder.convex_up.mask_pred.3.weight",
                "decoder.convex_up.mask_pred.3.bias");
            auxiliary_ = weights.value;
        } else {
            Tensor alpha = conv(fhalf, "decoder.convex_up.where_conv.0.weight", nullptr);
            alpha = batch_norm(alpha, "decoder.convex_up.where_conv.1", true);
            alpha = conv(alpha, "decoder.convex_up.where_conv.3.weight",
                nullptr, 1, 2, 1, alpha.channels);
            alpha = batch_norm(alpha, "decoder.convex_up.where_conv.4", true);
            alpha = conv(alpha, "decoder.convex_up.where_conv.6.weight", nullptr);
            alpha = resize(alpha, height_, width_, MPSGraphResizeBilinear, false);
            auxiliary_ = [graph_ sigmoidWithTensor:alpha.value name:nil];
        }
        if (fp16_) {
            depth_ = [graph_ castTensor:depth_ toType:MPSDataTypeFloat32 name:nil];
            auxiliary_ = [graph_ castTensor:auxiliary_
                toType:MPSDataTypeFloat32 name:nil];
        }
    }

private:
    MPSGraphTensor* constant(const std::string& name) {
        const TensorView& tensor = model_.tensor(name);
        NSData* data = [NSData dataWithBytesNoCopy:const_cast<float*>(tensor.data)
            length:tensor.elements * sizeof(float) freeWhenDone:NO];
        MPSGraphTensor* result = [graph_ constantWithData:data
            shape:shape(tensor) dataType:MPSDataTypeFloat32];
        return fp16_ ? [graph_ castTensor:result
            toType:MPSDataTypeFloat16 name:nil] : result;
    }
    MPSGraphTensor* channel_constant(const std::string& name) {
        const auto& tensor = model_.tensor(name);
        return [graph_ reshapeTensor:constant(name) withShape:
            shape({1, static_cast<NSInteger>(tensor.elements), 1, 1}) name:nil];
    }
    MPSGraphTensor* scalar(float value) {
        MPSGraphTensor* result = [graph_ constantWithScalar:value
            dataType:MPSDataTypeFloat32];
        return fp16_ ? [graph_ castTensor:result
            toType:MPSDataTypeFloat16 name:nil] : result;
    }
    MPSGraphTensor* add_value(MPSGraphTensor* a, MPSGraphTensor* b) {
        return [graph_ additionWithPrimaryTensor:a secondaryTensor:b name:nil];
    }
    MPSGraphTensor* multiply_value(MPSGraphTensor* a, MPSGraphTensor* b) {
        return [graph_ multiplicationWithPrimaryTensor:a secondaryTensor:b name:nil];
    }
    MPSGraphConvolution2DOpDescriptor* descriptor(
        int stride, int padding, int dilation, int groups) {
        return [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:stride
            strideInY:stride dilationRateInX:dilation dilationRateInY:dilation
            groups:groups paddingLeft:padding paddingRight:padding
            paddingTop:padding paddingBottom:padding
            paddingStyle:MPSGraphPaddingStyleExplicit
            dataLayout:MPSGraphTensorNamedDataLayoutNCHW
            weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
    }
    Tensor conv(const Tensor& input, const std::string& weight_name,
                const char* bias_name, int stride = 1, int padding = 0,
                int dilation = 1, int groups = 1) {
        const auto& weight = model_.tensor(weight_name);
        const int channels = static_cast<int>(weight.dimensions[0]);
        const int kh = static_cast<int>(weight.dimensions[2]);
        const int kw = static_cast<int>(weight.dimensions[3]);
        const int h = (input.height + 2 * padding - dilation * (kh - 1) - 1) /
            stride + 1;
        const int w = (input.width + 2 * padding - dilation * (kw - 1) - 1) /
            stride + 1;
        MPSGraphTensor* result = [graph_ convolution2DWithSourceTensor:input.value
            weightsTensor:constant(weight_name)
            descriptor:descriptor(stride, padding, dilation, groups) name:nil];
        if (bias_name) result = add_value(result, channel_constant(bias_name));
        return {result, channels, h, w};
    }
    Tensor batch_norm(const Tensor& input, const std::string& p, bool relu) {
        MPSGraphTensor* variance = add_value(
            channel_constant(p + ".running_var"), scalar(1.0e-5f));
        MPSGraphTensor* scale = [graph_ divisionWithPrimaryTensor:
            channel_constant(p + ".weight") secondaryTensor:
            [graph_ squareRootWithTensor:variance name:nil] name:nil];
        MPSGraphTensor* shift = [graph_ subtractionWithPrimaryTensor:
            channel_constant(p + ".bias") secondaryTensor:multiply_value(
                channel_constant(p + ".running_mean"), scale) name:nil];
        MPSGraphTensor* result = add_value(multiply_value(input.value, scale), shift);
        if (relu) result = [graph_ reLUWithTensor:result name:nil];
        return {result, input.channels, input.height, input.width};
    }
    Tensor conv_bn(const Tensor& input, const std::string& p, int stride = 1,
                   bool relu = true) {
        const auto& weight = model_.tensor(p + ".conv.weight");
        return batch_norm(conv(input, p + ".conv.weight", nullptr, stride,
            static_cast<int>(weight.dimensions[2] / 2)), p + ".bn", relu);
    }
    Tensor add(const Tensor& a, const Tensor& b, float scale = 1.0f) {
        MPSGraphTensor* right = scale == 1.0f ? b.value :
            multiply_value(b.value, scalar(scale));
        return {add_value(a.value, right), a.channels, a.height, a.width};
    }
    Tensor rep(const Tensor& input, const std::string& p, int stride = 1) {
        Tensor a = batch_norm(conv(input, p + ".branch_3x3.0.weight", nullptr,
            stride, 1), p + ".branch_3x3.1", false);
        Tensor b = batch_norm(conv(input, p + ".branch_1x1.0.weight", nullptr,
            stride, 0), p + ".branch_1x1.1", false);
        Tensor result = add(a, b);
        if (stride == 1 && input.channels == result.channels)
            result = add(result, input);
        result.value = [graph_ reLUWithTensor:result.value name:nil];
        return result;
    }
    Tensor resize(const Tensor& input, int h, int w, MPSGraphResizeMode mode,
                  bool align_corners) {
        return {[graph_ resizeTensor:input.value size:shape({h, w}) mode:mode
            centerResult:align_corners ? NO : YES
            alignCorners:align_corners ? YES : NO
            layout:MPSGraphTensorNamedDataLayoutNCHW name:nil],
            input.channels, h, w};
    }
    MPSGraphPooling4DOpDescriptor* pool_descriptor(
        int kernel, int stride, int padding) {
        return [MPSGraphPooling4DOpDescriptor
            descriptorWithKernelSizes:@[@1, @1, @(kernel), @(kernel)]
            strides:@[@1, @1, @(stride), @(stride)]
            dilationRates:@[@1, @1, @1, @1]
            paddingValues:@[@0, @0, @0, @0, @(padding), @(padding),
                @(padding), @(padding)]
            paddingStyle:MPSGraphPaddingStyleExplicit];
    }
    Tensor max_pool5(const Tensor& input) {
        MPSGraphTensor* padded = [graph_ padTensor:input.value
            withPaddingMode:MPSGraphPaddingModeClampToEdge
            leftPadding:shape({0, 0, 2, 2}) rightPadding:shape({0, 0, 2, 2})
            constantValue:0.0 name:nil];
        return {[graph_ maxPooling4DWithSourceTensor:padded
            descriptor:pool_descriptor(5, 1, 0) name:nil],
            input.channels, input.height, input.width};
    }
    Tensor average_pool2(const Tensor& input) {
        return {[graph_ avgPooling4DWithSourceTensor:input.value
            descriptor:pool_descriptor(2, 2, 0) name:nil],
            input.channels, input.height / 2, input.width / 2};
    }
    Tensor fusion(const Tensor& high, const Tensor& low, const std::string& p) {
        const int high_groups = high.channels % 4 == 0 &&
            model_.tensor(p + ".proj_high.weight").dimensions[0] % 4 == 0 ? 4 : 1;
        const int low_groups = low.channels % 4 == 0 &&
            model_.tensor(p + ".proj_low.weight").dimensions[0] % 4 == 0 ? 4 : 1;
        Tensor h = conv(high, p + ".proj_high.weight", nullptr, 1, 0, 1, high_groups);
        Tensor l = conv(low, p + ".proj_low.weight", nullptr, 1, 0, 1, low_groups);
        l = resize(l, high.height, high.width, MPSGraphResizeBilinear, false);
        return batch_norm(add(h, l), p + ".bn", true);
    }

    const ModelFile& model_;
    int width_, height_;
    bool fp16_;
    MPSGraph* graph_ = nil;
    MPSGraphTensor* input_ = nil;
    MPSGraphTensor* depth_ = nil;
    MPSGraphTensor* auxiliary_ = nil;
};

struct Plan {
    MPSGraph* graph = nil;
    MPSGraphTensor* input = nil;
    MPSGraphExecutable* executable = nil;
};

}  // namespace

class MetalExecutor::Impl {
public:
    explicit Impl(const std::string& path) : model_(path) {
        device_ = MTLCreateSystemDefaultDevice();
        queue_ = [device_ newCommandQueue];
        graph_device_ = [MPSGraphDevice deviceWithMTLDevice:device_];
        if (device_ == nil || queue_ == nil || graph_device_ == nil)
            throw std::runtime_error("Metal is unavailable for ZipDepth");
        const auto precision = inferbridge::native::requested_precision();
        if (precision == inferbridge::native::Precision::int8)
            throw std::invalid_argument("ZipDepth Metal does not support INT8");
        fp16_ = precision == inferbridge::native::Precision::fp16 ||
            precision == inferbridge::native::Precision::automatic;
    }
    void infer(const float* rgb, std::uint32_t width, std::uint32_t height,
               float* output, std::uint64_t elements) {
        if (!rgb || !output || !width || !height || width % 32 || height % 32 ||
            elements < static_cast<std::uint64_t>(width) * height)
            throw std::invalid_argument("invalid ZipDepth Metal tensor shape");
        std::lock_guard<std::mutex> lock(mutex_);
        @autoreleasepool {
            Plan& plan = get_plan(width, height);
            id<MTLBuffer> buffer = [device_ newBufferWithBytes:rgb
                length:static_cast<NSUInteger>(width) * height * 3u * sizeof(float)
                options:MTLResourceStorageModeShared];
            MPSGraphTensorData* data = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:buffer shape:shape({1, 3,
                    static_cast<NSInteger>(height), static_cast<NSInteger>(width)})
                dataType:MPSDataTypeFloat32];
            MPSGraphExecutableExecutionDescriptor* descriptor =
                [MPSGraphExecutableExecutionDescriptor new];
            descriptor.waitUntilCompleted = YES;
            NSArray<MPSGraphTensorData*>* results = [plan.executable
                runWithMTLCommandQueue:queue_ inputsArray:@[data]
                resultsArray:nil executionDescriptor:descriptor];
            if (results.count != 2u)
                throw std::runtime_error("ZipDepth Metal graph returned invalid outputs");
            const std::size_t low_width = width / 2u;
            const std::size_t low_height = height / 2u;
            const std::size_t low_count = low_width * low_height;
            std::vector<float> depth(low_count);
            [results[0].mpsndarray readBytes:depth.data() strideBytes:nil];
            if (model_.kind() == ZIPDEPTH_MODEL_BASE_GPU) {
                std::vector<float> weights(low_count * 36u);
                [results[1].mpsndarray readBytes:weights.data() strideBytes:nil];
                for (std::uint32_t y = 0; y < height; ++y)
                    for (std::uint32_t x = 0; x < width; ++x) {
                        const std::uint32_t py = y / 2u, px = x / 2u;
                        const std::uint32_t subpixel = (y % 2u) * 2u + x % 2u;
                        float maximum = -std::numeric_limits<float>::infinity();
                        for (std::uint32_t n = 0; n < 9; ++n)
                            maximum = std::max(maximum, weights[
                                (static_cast<std::size_t>(n * 4u + subpixel) *
                                    low_height + py) * low_width + px]);
                        float total = 0.0f, value = 0.0f;
                        for (std::uint32_t n = 0; n < 9; ++n) {
                            const float weight = std::exp(weights[
                                (static_cast<std::size_t>(n * 4u + subpixel) *
                                    low_height + py) * low_width + px] - maximum);
                            total += weight;
                            const int ny = std::clamp<int>(
                                static_cast<int>(py) + static_cast<int>(n / 3u) - 1,
                                0, static_cast<int>(low_height) - 1);
                            const int nx = std::clamp<int>(
                                static_cast<int>(px) + static_cast<int>(n % 3u) - 1,
                                0, static_cast<int>(low_width) - 1);
                            value += weight * depth[
                                static_cast<std::size_t>(ny) * low_width + nx];
                        }
                        output[static_cast<std::size_t>(y) * width + x] =
                            std::max(value / total, 0.0f);
                    }
            } else {
                std::vector<float> alpha(static_cast<std::size_t>(width) * height);
                [results[1].mpsndarray readBytes:alpha.data() strideBytes:nil];
                for (std::uint32_t y = 0; y < height; ++y)
                    for (std::uint32_t x = 0; x < width; ++x) {
                        const float source_y = (y + 0.5f) * low_height / height - 0.5f;
                        const float source_x = (x + 0.5f) * low_width / width - 0.5f;
                        const int y0 = std::clamp<int>(std::floor(source_y), 0, low_height - 1);
                        const int x0 = std::clamp<int>(std::floor(source_x), 0, low_width - 1);
                        const int y1 = std::min<int>(y0 + 1, low_height - 1);
                        const int x1 = std::min<int>(x0 + 1, low_width - 1);
                        const float fy = std::clamp(source_y, 0.0f,
                            static_cast<float>(low_height - 1)) - y0;
                        const float fx = std::clamp(source_x, 0.0f,
                            static_cast<float>(low_width - 1)) - x0;
                        const float top = depth[y0 * low_width + x0] * (1 - fx) +
                            depth[y0 * low_width + x1] * fx;
                        const float bottom = depth[y1 * low_width + x0] * (1 - fx) +
                            depth[y1 * low_width + x1] * fx;
                        const float bilinear = top * (1 - fy) + bottom * fy;
                        const float nearest = depth[(y / 2u) * low_width + x / 2u];
                        const std::size_t index = static_cast<std::size_t>(y) * width + x;
                        output[index] = std::max(
                            alpha[index] * nearest + (1.0f - alpha[index]) * bilinear,
                            0.0f);
                    }
            }
        }
    }
private:
    Plan& get_plan(int width, int height) {
        const std::uint64_t key = (static_cast<std::uint64_t>(width) << 32u) |
            static_cast<std::uint32_t>(height);
        auto found = plans_.find(key);
        if (found != plans_.end()) return found->second;
        GraphBuilder builder(model_, width, height, fp16_);
        builder.build();
        MPSGraphShapedType* type = [[MPSGraphShapedType alloc]
            initWithShape:shape({1, 3, height, width}) dataType:MPSDataTypeFloat32];
        MPSGraphCompilationDescriptor* descriptor = [MPSGraphCompilationDescriptor new];
        descriptor.optimizationLevel = MPSGraphOptimizationLevel1;
        descriptor.waitForCompilationCompletion = YES;
        MPSGraphExecutable* executable = [builder.graph() compileWithDevice:graph_device_
            feeds:@{builder.input(): type} targetTensors:builder.outputs()
            targetOperations:nil compilationDescriptor:descriptor];
        if (executable == nil)
            throw std::runtime_error("failed to compile ZipDepth Metal graph");
        executable.options = MPSGraphOptionsSynchronizeResults;
        return plans_.emplace(key,
            Plan{builder.graph(), builder.input(), executable}).first->second;
    }
    ModelFile model_;
    bool fp16_ = false;
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    MPSGraphDevice* graph_device_ = nil;
    std::unordered_map<std::uint64_t, Plan> plans_;
    std::mutex mutex_;
};

MetalExecutor::MetalExecutor(const std::string& path)
    : impl_(std::make_unique<Impl>(path)) {}
MetalExecutor::~MetalExecutor() = default;
void MetalExecutor::infer(const float* rgb, std::uint32_t width,
                          std::uint32_t height, float* depth,
                          std::uint64_t elements) {
    impl_->infer(rgb, width, height, depth, elements);
}

}  // namespace zipdepth_native
