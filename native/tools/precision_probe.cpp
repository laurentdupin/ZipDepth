#include "vulkan.h"

#include "precision_pointwise_fp16_spv.h"
#include "precision_pointwise_fp16_weights_spv.h"
#include "precision_pointwise_fp32_spv.h"
#include "precision_pointwise_int8_spv.h"
#include "quantize_rows_int8_spv.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kSpatial = 4096;
constexpr std::uint32_t kInputChannels = 128;
constexpr std::uint32_t kOutputChannels = 128;
constexpr std::uint32_t kWarmup = 5;
constexpr std::uint32_t kIterations = 20;

struct Parameters {
    std::uint32_t spatial;
    std::uint32_t input_channels;
    std::uint32_t output_channels;
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
    if (exponent == 0xffu) {
        return static_cast<std::uint16_t>(
            sign | 0x7c00u | (mantissa != 0u ? 0x0200u : 0u));
    }
    const int adjusted = static_cast<int>(exponent) - 127 + 15;
    if (adjusted >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
    if (adjusted <= 0) {
        if (adjusted < -10) return static_cast<std::uint16_t>(sign);
        const std::uint32_t hidden = mantissa | 0x800000u;
        const std::uint32_t shift = static_cast<std::uint32_t>(14 - adjusted);
        const std::uint32_t rounded =
            (hidden + (1u << (shift - 1u))) >> shift;
        return static_cast<std::uint16_t>(sign | rounded);
    }
    const std::uint32_t rounded = mantissa + 0x1000u;
    if ((rounded & 0x800000u) != 0u) {
        if (adjusted + 1 >= 31)
            return static_cast<std::uint16_t>(sign | 0x7c00u);
        return static_cast<std::uint16_t>(
            sign | (static_cast<std::uint32_t>(adjusted + 1) << 10u));
    }
    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint32_t>(adjusted) << 10u) |
        (rounded >> 13u));
}

std::uint32_t pack_signed_4(const std::int8_t* values) {
    return static_cast<std::uint32_t>(
        static_cast<std::uint8_t>(values[0])) |
        (static_cast<std::uint32_t>(
            static_cast<std::uint8_t>(values[1])) << 8u) |
        (static_cast<std::uint32_t>(
            static_cast<std::uint8_t>(values[2])) << 16u) |
        (static_cast<std::uint32_t>(
            static_cast<std::uint8_t>(values[3])) << 24u);
}

struct ErrorMetrics {
    double maximum_absolute = 0.0;
    double rmse = 0.0;
    double relative_rmse = 0.0;
};

ErrorMetrics sampled_error(
    const std::vector<float>& actual,
    const std::vector<float>& input,
    const std::vector<float>& weight) {
    double squared_error = 0.0;
    double squared_reference = 0.0;
    std::uint64_t samples = 0;
    for (std::uint32_t position = 0; position < kSpatial; position += 31u) {
        for (std::uint32_t output = 0;
             output < kOutputChannels; output += 7u) {
            float reference = 0.0f;
            for (std::uint32_t input_channel = 0;
                 input_channel < kInputChannels; ++input_channel) {
                reference += input[
                    position * kInputChannels + input_channel] *
                    weight[output * kInputChannels + input_channel];
            }
            const float difference = actual[
                position * kOutputChannels + output] - reference;
            squared_error += static_cast<double>(difference) * difference;
            squared_reference += static_cast<double>(reference) * reference;
            ++samples;
        }
    }
    ErrorMetrics result;
    result.rmse = std::sqrt(squared_error / static_cast<double>(samples));
    result.relative_rmse = std::sqrt(
        squared_error / std::max(squared_reference, 1.0e-20));
    for (std::uint32_t position = 0; position < kSpatial; position += 31u) {
        for (std::uint32_t output = 0;
             output < kOutputChannels; output += 7u) {
            float reference = 0.0f;
            for (std::uint32_t input_channel = 0;
                 input_channel < kInputChannels; ++input_channel) {
                reference += input[
                    position * kInputChannels + input_channel] *
                    weight[output * kInputChannels + input_channel];
            }
            result.maximum_absolute = std::max(
                result.maximum_absolute,
                std::abs(static_cast<double>(actual[
                    position * kOutputChannels + output] - reference)));
        }
    }
    return result;
}

double benchmark(
    midas_native::VulkanContext& context,
    const midas_native::VulkanPipeline& pipeline,
    const std::vector<const midas_native::VulkanBuffer*>& buffers,
    const Parameters& parameters) {
    const auto run = [&] {
        context.dispatch(
            pipeline, buffers, &parameters, sizeof(parameters),
            divide_up(kSpatial * kOutputChannels, 64u));
    };
    for (std::uint32_t index = 0; index < kWarmup; ++index) run();
    const auto started = std::chrono::steady_clock::now();
    context.batch([&] {
        for (std::uint32_t index = 0; index < kIterations; ++index) run();
    });
    const auto finished = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(
        finished - started).count() / static_cast<double>(kIterations);
}

void print_result(
    const char* name, double milliseconds, const ErrorMetrics& error) {
    std::cout << name << "_ms=" << std::fixed << std::setprecision(3)
              << milliseconds << " " << name << "_gmacs="
              << (static_cast<double>(kSpatial) * kInputChannels *
                  kOutputChannels / 1.0e6 / milliseconds)
              << " " << name << "_max_abs=" << std::scientific
              << error.maximum_absolute << " " << name << "_rel_rmse="
              << error.relative_rmse << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::uint32_t device_index = argc > 1
            ? static_cast<std::uint32_t>(std::stoul(argv[1])) : 0u;
        midas_native::VulkanContext context(device_index, true);
        std::cout << "device=\"" << context.device_name() << "\""
                  << " fp16=" << context.supports_float16()
                  << " packed_int8_dot="
                  << context.supports_packed_int8_dot() << "\n";
        if (!context.supports_float16() &&
            !context.supports_packed_int8_dot())
            throw std::runtime_error("no reduced-precision feature is available");

        std::mt19937 random(0x5a17u);
        std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
        std::vector<float> input(
            static_cast<std::size_t>(kSpatial) * kInputChannels);
        std::vector<float> weight(
            static_cast<std::size_t>(kOutputChannels) * kInputChannels);
        for (float& value : input) value = distribution(random);
        for (float& value : weight) value = distribution(random) * 0.2f;
        std::vector<float> output(
            static_cast<std::size_t>(kSpatial) * kOutputChannels);
        const Parameters parameters{
            kSpatial, kInputChannels, kOutputChannels};

        auto fp32_output = context.create_device_buffer(
            output.size() * sizeof(float));
        auto fp32_input = context.create_device_buffer(
            input.size() * sizeof(float));
        auto fp32_weight = context.create_device_buffer(
            weight.size() * sizeof(float));
        context.upload(fp32_input, input.data(), input.size() * sizeof(float));
        context.upload(
            fp32_weight, weight.data(), weight.size() * sizeof(float));
        auto fp32_pipeline = context.create_pipeline(
            midas_precision_pointwise_fp32_spv,
            midas_precision_pointwise_fp32_spv_size, 3u,
            sizeof(Parameters));
        const double fp32_ms = benchmark(
            context, fp32_pipeline,
            {&fp32_output, &fp32_input, &fp32_weight}, parameters);
        context.download(
            fp32_output, output.data(), output.size() * sizeof(float));
        print_result("fp32", fp32_ms, sampled_error(output, input, weight));

        double fp16_ms = std::numeric_limits<double>::quiet_NaN();
        double fp16_weight_ms = std::numeric_limits<double>::quiet_NaN();
        if (context.supports_float16()) {
            std::vector<std::uint16_t> input_half(input.size());
            std::vector<std::uint16_t> weight_half(weight.size());
            std::transform(input.begin(), input.end(), input_half.begin(),
                float_to_half);
            std::transform(weight.begin(), weight.end(), weight_half.begin(),
                float_to_half);
            auto fp16_output = context.create_device_buffer(
                output.size() * sizeof(float));
            auto fp16_input = context.create_device_buffer(
                input_half.size() * sizeof(std::uint16_t));
            auto fp16_weight = context.create_device_buffer(
                weight_half.size() * sizeof(std::uint16_t));
            context.upload(fp16_input, input_half.data(),
                input_half.size() * sizeof(std::uint16_t));
            context.upload(fp16_weight, weight_half.data(),
                weight_half.size() * sizeof(std::uint16_t));
            auto fp16_pipeline = context.create_pipeline(
                midas_precision_pointwise_fp16_spv,
                midas_precision_pointwise_fp16_spv_size, 3u,
                sizeof(Parameters));
            fp16_ms = benchmark(context, fp16_pipeline,
                {&fp16_output, &fp16_input, &fp16_weight}, parameters);
            context.download(fp16_output, output.data(),
                output.size() * sizeof(float));
            print_result("fp16", fp16_ms,
                sampled_error(output, input, weight));

            auto fp16_weight_output = context.create_device_buffer(
                output.size() * sizeof(float));
            auto fp16_weight_pipeline = context.create_pipeline(
                midas_precision_pointwise_fp16_weights_spv,
                midas_precision_pointwise_fp16_weights_spv_size, 3u,
                sizeof(Parameters));
            fp16_weight_ms = benchmark(context, fp16_weight_pipeline,
                {&fp16_weight_output, &fp32_input, &fp16_weight}, parameters);
            context.download(fp16_weight_output, output.data(),
                output.size() * sizeof(float));
            print_result("fp16_weight", fp16_weight_ms,
                sampled_error(output, input, weight));
        } else {
            std::cout << "fp16=unsupported fp16_weight=unsupported\n";
        }

        const std::uint32_t packed_channels = kInputChannels / 4u;
        std::vector<float> input_scale(kSpatial);
        std::vector<float> weight_scale(kOutputChannels);
        std::vector<std::uint32_t> input_packed(
            static_cast<std::size_t>(kSpatial) * packed_channels);
        std::vector<std::uint32_t> weight_packed(
            static_cast<std::size_t>(kOutputChannels) * packed_channels);
        for (std::uint32_t position = 0; position < kSpatial; ++position) {
            float maximum = 0.0f;
            for (std::uint32_t channel = 0;
                 channel < kInputChannels; ++channel)
                maximum = std::max(maximum, std::abs(input[
                    position * kInputChannels + channel]));
            input_scale[position] = std::max(maximum / 127.0f, 1.0e-8f);
            for (std::uint32_t group = 0;
                 group < packed_channels; ++group) {
                std::int8_t values[4]{};
                for (std::uint32_t lane = 0; lane < 4u; ++lane)
                    values[lane] = static_cast<std::int8_t>(std::clamp(
                        std::lround(input[position * kInputChannels +
                            group * 4u + lane] / input_scale[position]),
                        -127l, 127l));
                input_packed[position * packed_channels + group] =
                    pack_signed_4(values);
            }
        }
        for (std::uint32_t output_channel = 0;
             output_channel < kOutputChannels; ++output_channel) {
            float maximum = 0.0f;
            for (std::uint32_t channel = 0;
                 channel < kInputChannels; ++channel)
                maximum = std::max(maximum, std::abs(weight[
                    output_channel * kInputChannels + channel]));
            weight_scale[output_channel] =
                std::max(maximum / 127.0f, 1.0e-8f);
            for (std::uint32_t group = 0;
                 group < packed_channels; ++group) {
                std::int8_t values[4]{};
                for (std::uint32_t lane = 0; lane < 4u; ++lane)
                    values[lane] = static_cast<std::int8_t>(std::clamp(
                        std::lround(weight[output_channel * kInputChannels +
                            group * 4u + lane] /
                            weight_scale[output_channel]), -127l, 127l));
                weight_packed[output_channel * packed_channels + group] =
                    pack_signed_4(values);
            }
        }
        auto int8_output = context.create_device_buffer(
            output.size() * sizeof(float));
        auto int8_input = context.create_device_buffer(
            input_packed.size() * sizeof(std::uint32_t));
        auto int8_weight = context.create_device_buffer(
            weight_packed.size() * sizeof(std::uint32_t));
        auto int8_input_scale = context.create_device_buffer(
            input_scale.size() * sizeof(float));
        auto int8_weight_scale = context.create_device_buffer(
            weight_scale.size() * sizeof(float));
        context.upload(
            int8_input, input_packed.data(),
            input_packed.size() * sizeof(std::uint32_t));
        context.upload(
            int8_weight, weight_packed.data(),
            weight_packed.size() * sizeof(std::uint32_t));
        context.upload(
            int8_input_scale, input_scale.data(),
            input_scale.size() * sizeof(float));
        context.upload(
            int8_weight_scale, weight_scale.data(),
            weight_scale.size() * sizeof(float));
        auto int8_pipeline = context.create_pipeline(
            midas_precision_pointwise_int8_spv,
            midas_precision_pointwise_int8_spv_size, 5u,
            sizeof(Parameters));
        const double int8_ms = benchmark(
            context, int8_pipeline,
            {&int8_output, &int8_input, &int8_weight,
             &int8_input_scale, &int8_weight_scale}, parameters);
        context.download(
            int8_output, output.data(), output.size() * sizeof(float));
        print_result("int8", int8_ms, sampled_error(output, input, weight));
        auto dynamic_input = context.create_device_buffer(
            input_packed.size() * sizeof(std::uint32_t));
        auto dynamic_scale = context.create_device_buffer(
            input_scale.size() * sizeof(float));
        auto quantize_pipeline = context.create_pipeline(
            midas_quantize_rows_int8_spv,
            midas_quantize_rows_int8_spv_size, 3u, 8u);
        const std::uint32_t quantize_parameters[2]{
            kSpatial, kInputChannels};
        const auto dynamic_run = [&] {
            context.dispatch(quantize_pipeline,
                {&dynamic_input, &dynamic_scale, &fp32_input},
                quantize_parameters, sizeof(quantize_parameters), kSpatial);
            context.dispatch(int8_pipeline,
                {&int8_output, &dynamic_input, &int8_weight,
                 &dynamic_scale, &int8_weight_scale},
                &parameters, sizeof(parameters),
                divide_up(kSpatial * kOutputChannels, 64u));
        };
        for (std::uint32_t index = 0; index < kWarmup; ++index)
            dynamic_run();
        const auto dynamic_started = std::chrono::steady_clock::now();
        context.batch([&] {
            for (std::uint32_t index = 0; index < kIterations; ++index)
                dynamic_run();
        });
        const double dynamic_int8_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - dynamic_started).count() /
            static_cast<double>(kIterations);
        context.download(
            int8_output, output.data(), output.size() * sizeof(float));
        print_result("dynamic_int8", dynamic_int8_ms,
            sampled_error(output, input, weight));
        if (context.supports_float16())
            std::cout << "fp16_speedup=" << std::fixed << std::setprecision(3)
                      << fp32_ms / fp16_ms << " fp16_weight_speedup="
                      << fp32_ms / fp16_weight_ms << " ";
        std::cout << "int8_speedup=" << std::fixed << std::setprecision(3)
                  << fp32_ms / int8_ms << " dynamic_int8_speedup="
                  << fp32_ms / dynamic_int8_ms << "\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "precision_probe_failed=\"" << exception.what()
                  << "\"\n";
        return 1;
    }
}
