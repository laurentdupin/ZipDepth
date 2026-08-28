#include "vulkan.h"
#include "vulkan_operators.h"

#include "precision_spatial_fp16_spv.h"
#include "precision_spatial_int8_spv.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::uint32_t kWidth = 96;
constexpr std::uint32_t kHeight = 64;
constexpr std::uint32_t kInputChannels = 64;
constexpr std::uint32_t kOutputChannels = 64;
constexpr std::uint32_t kWarmup = 3;
constexpr std::uint32_t kIterations = 10;

struct ConvParameters {
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
};

struct Int8Parameters {
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t input_channels;
    std::uint32_t output_channels;
    float input_scale;
};

std::uint32_t divide_up(std::uint32_t value, std::uint32_t divisor) {
    return (value + divisor - 1u) / divisor;
}

std::uint16_t float_to_half(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16u) & 0x8000u;
    const std::uint32_t exponent = (bits >> 23u) & 0xffu;
    const std::uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu)
        return static_cast<std::uint16_t>(
            sign | 0x7c00u | (mantissa ? 0x0200u : 0u));
    const int adjusted = static_cast<int>(exponent) - 112;
    if (adjusted >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
    if (adjusted <= 0) {
        if (adjusted < -10) return static_cast<std::uint16_t>(sign);
        const std::uint32_t hidden = mantissa | 0x800000u;
        const std::uint32_t shift = static_cast<std::uint32_t>(14 - adjusted);
        return static_cast<std::uint16_t>(
            sign | ((hidden + (1u << (shift - 1u))) >> shift));
    }
    const std::uint32_t rounded = mantissa + 0x1000u;
    if (rounded & 0x800000u)
        return static_cast<std::uint16_t>(
            sign | (static_cast<std::uint32_t>(adjusted + 1) << 10u));
    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint32_t>(adjusted) << 10u) |
        (rounded >> 13u));
}

std::uint32_t pack4(const std::int8_t* values) {
    std::uint32_t result = 0;
    for (std::uint32_t lane = 0; lane < 4; ++lane)
        result |= static_cast<std::uint32_t>(
            static_cast<std::uint8_t>(values[lane])) << (lane * 8u);
    return result;
}

template <typename Function>
double benchmark(midas_native::VulkanContext& context, Function&& function) {
    for (std::uint32_t index = 0; index < kWarmup; ++index) function();
    const auto start = std::chrono::steady_clock::now();
    context.batch([&] {
        for (std::uint32_t index = 0; index < kIterations; ++index)
            function();
    });
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count() / kIterations;
}

struct Error {
    double maximum = 0.0;
    double relative_rmse = 0.0;
};

Error error(
    const std::vector<float>& output,
    const std::vector<float>& input,
    const std::vector<float>& weight) {
    double squared_error = 0.0;
    double squared_reference = 0.0;
    Error result;
    for (std::uint32_t oc = 0; oc < kOutputChannels; oc += 5u)
        for (std::uint32_t y = 0; y < kHeight; y += 7u)
            for (std::uint32_t x = 0; x < kWidth; x += 11u) {
                float reference = 0.0f;
                for (std::uint32_t ic = 0; ic < kInputChannels; ++ic)
                    for (std::uint32_t ky = 0; ky < 3; ++ky)
                        for (std::uint32_t kx = 0; kx < 3; ++kx) {
                            const int iy = static_cast<int>(y + ky) - 1;
                            const int ix = static_cast<int>(x + kx) - 1;
                            if (iy < 0 || ix < 0 || iy >= int(kHeight) ||
                                ix >= int(kWidth)) continue;
                            reference += input[
                                (ic * kHeight + std::uint32_t(iy)) * kWidth +
                                std::uint32_t(ix)] * weight[
                                ((oc * kInputChannels + ic) * 3 + ky) * 3 +
                                kx];
                        }
                const double difference = output[
                    (oc * kHeight + y) * kWidth + x] - reference;
                result.maximum = std::max(result.maximum,
                    std::abs(difference));
                squared_error += difference * difference;
                squared_reference += double(reference) * reference;
            }
    result.relative_rmse =
        std::sqrt(squared_error / std::max(squared_reference, 1.0e-20));
    return result;
}

void print(const char* name, double ms, const Error& value) {
    const double gmac = double(kWidth) * kHeight * kInputChannels *
        kOutputChannels * 9.0 / 1.0e9;
    std::cout << name << "_ms=" << std::fixed << std::setprecision(3) << ms
              << " " << name << "_gmacs=" << gmac * 1000.0 / ms
              << " " << name << "_max_abs=" << std::scientific
              << value.maximum << " " << name << "_rel_rmse="
              << value.relative_rmse << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::uint32_t device_index = argc > 1
            ? static_cast<std::uint32_t>(std::stoul(argv[1])) : 0u;
        midas_native::VulkanContext context(device_index, true);
        midas_native::VulkanOperators operators(context);
        std::cout << "device=\"" << context.device_name() << "\" fp16="
                  << context.supports_float16() << " packed_int8_dot="
                  << context.supports_packed_int8_dot() << "\n";
        std::mt19937 random(0x7319u);
        std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
        const std::size_t input_count =
            std::size_t(kWidth) * kHeight * kInputChannels;
        const std::size_t weight_count =
            std::size_t(kOutputChannels) * kInputChannels * 9u;
        const std::size_t output_count =
            std::size_t(kWidth) * kHeight * kOutputChannels;
        std::vector<float> input(input_count), weight(weight_count);
        std::vector<float> output(output_count), zeros(kOutputChannels, 0.0f);
        for (float& value : input) value = distribution(random);
        for (float& value : weight) value = distribution(random) * 0.05f;

        auto fp32_input = context.create_device_buffer(input.size() * 4u);
        auto fp32_weight = context.create_device_buffer(weight.size() * 4u);
        auto bias = context.create_device_buffer(zeros.size() * 4u);
        auto fp32_output = context.create_device_buffer(output.size() * 4u);
        context.upload(fp32_input, input.data(), input.size() * 4u);
        context.upload(fp32_weight, weight.data(), weight.size() * 4u);
        context.upload(bias, zeros.data(), zeros.size() * 4u);
        const double fp32_ms = benchmark(context, [&] {
            operators.conv(fp32_output, fp32_input, fp32_weight, bias,
                kWidth, kHeight, kInputChannels, kWidth, kHeight,
                kOutputChannels, 3u, 3u, 1u, 1, 1, 1u, 1u, false,
                nullptr, nullptr, nullptr, nullptr, 0u, false, nullptr);
        });
        context.download(fp32_output, output.data(), output.size() * 4u);
        print("fp32_spatial", fp32_ms, error(output, input, weight));

        std::vector<std::uint16_t> input_half(input.size());
        std::vector<std::uint16_t> weight_half(weight.size());
        std::transform(input.begin(), input.end(), input_half.begin(),
            float_to_half);
        std::transform(weight.begin(), weight.end(), weight_half.begin(),
            float_to_half);
        auto fp16_input = context.create_device_buffer(input_half.size() * 2u);
        auto fp16_weight = context.create_device_buffer(weight_half.size() * 2u);
        auto fp16_output = context.create_device_buffer(output.size() * 4u);
        context.upload(fp16_input, input_half.data(), input_half.size() * 2u);
        context.upload(fp16_weight, weight_half.data(), weight_half.size() * 2u);
        const ConvParameters conv_parameters{
            kWidth, kHeight, kInputChannels, kWidth, kHeight,
            kOutputChannels, 3u, 3u, 1u, 1, 1, 1u, 1u, 0u, 0u, 0u,
            0.001f};
        auto fp16_pipeline = context.create_pipeline(
            midas_precision_spatial_fp16_spv,
            midas_precision_spatial_fp16_spv_size, 4u,
            sizeof(ConvParameters));
        const double fp16_ms = benchmark(context, [&] {
            context.dispatch(fp16_pipeline,
                {&fp16_output, &fp16_input, &fp16_weight, &bias},
                &conv_parameters, sizeof(conv_parameters),
                divide_up(kWidth, 16u), divide_up(kHeight, 4u),
                divide_up(kOutputChannels, 8u));
        });
        context.download(fp16_output, output.data(), output.size() * 4u);
        print("fp16_spatial", fp16_ms, error(output, input, weight));

        float input_maximum = 0.0f;
        for (float value : input)
            input_maximum = std::max(input_maximum, std::abs(value));
        const float input_scale = input_maximum / 127.0f;
        const std::uint32_t packed_channels = kInputChannels / 4u;
        std::vector<std::uint32_t> input_packed(
            std::size_t(kWidth) * kHeight * packed_channels);
        for (std::uint32_t y = 0; y < kHeight; ++y)
            for (std::uint32_t x = 0; x < kWidth; ++x)
                for (std::uint32_t group = 0;
                     group < packed_channels; ++group) {
                    std::int8_t values[4]{};
                    for (std::uint32_t lane = 0; lane < 4u; ++lane)
                        values[lane] = static_cast<std::int8_t>(std::clamp(
                            std::lround(input[
                                ((group * 4u + lane) * kHeight + y) *
                                kWidth + x] / input_scale), -127l, 127l));
                    input_packed[(y * kWidth + x) * packed_channels + group] =
                        pack4(values);
                }
        std::vector<float> weight_scale(kOutputChannels);
        std::vector<std::uint32_t> weight_packed(
            std::size_t(kOutputChannels) * 9u * packed_channels);
        for (std::uint32_t oc = 0; oc < kOutputChannels; ++oc) {
            float maximum = 0.0f;
            for (std::uint32_t ic = 0; ic < kInputChannels; ++ic)
                for (std::uint32_t kernel = 0; kernel < 9u; ++kernel)
                    maximum = std::max(maximum, std::abs(weight[
                        (oc * kInputChannels + ic) * 9u + kernel]));
            weight_scale[oc] = maximum / 127.0f;
            for (std::uint32_t kernel = 0; kernel < 9u; ++kernel)
                for (std::uint32_t group = 0;
                     group < packed_channels; ++group) {
                    std::int8_t values[4]{};
                    for (std::uint32_t lane = 0; lane < 4u; ++lane)
                        values[lane] = static_cast<std::int8_t>(std::clamp(
                            std::lround(weight[
                                (oc * kInputChannels + group * 4u + lane) *
                                9u + kernel] / weight_scale[oc]),
                            -127l, 127l));
                    weight_packed[(oc * 9u + kernel) * packed_channels +
                        group] = pack4(values);
                }
        }
        auto int8_input = context.create_device_buffer(input_packed.size() * 4u);
        auto int8_weight = context.create_device_buffer(weight_packed.size() * 4u);
        auto int8_scale = context.create_device_buffer(weight_scale.size() * 4u);
        auto int8_output = context.create_device_buffer(output.size() * 4u);
        context.upload(int8_input, input_packed.data(), input_packed.size() * 4u);
        context.upload(int8_weight, weight_packed.data(), weight_packed.size() * 4u);
        context.upload(int8_scale, weight_scale.data(), weight_scale.size() * 4u);
        const Int8Parameters int8_parameters{
            kWidth, kHeight, kInputChannels, kOutputChannels, input_scale};
        auto int8_pipeline = context.create_pipeline(
            midas_precision_spatial_int8_spv,
            midas_precision_spatial_int8_spv_size, 4u,
            sizeof(Int8Parameters));
        const double int8_ms = benchmark(context, [&] {
            context.dispatch(int8_pipeline,
                {&int8_output, &int8_input, &int8_weight, &int8_scale},
                &int8_parameters, sizeof(int8_parameters),
                divide_up(kWidth, 8u), divide_up(kHeight, 8u),
                kOutputChannels);
        });
        context.download(int8_output, output.data(), output.size() * 4u);
        print("int8_spatial", int8_ms, error(output, input, weight));
        std::cout << "fp16_spatial_speedup=" << std::fixed
                  << std::setprecision(3) << fp32_ms / fp16_ms
                  << " int8_spatial_speedup=" << fp32_ms / int8_ms << "\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "spatial_precision_probe_failed=\"" << exception.what()
                  << "\"\n";
        return 1;
    }
}
