#include "vulkan.h"
#include "add_spv.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    setenv("ZIPDEPTH_ANDROID_BATCH_DISPATCH_LIMIT", "4", 1);
    try {
        midas_native::VulkanContext context(0);
        auto pipeline = context.create_pipeline(midas_add_spv, midas_add_spv_size, 3, 4);
        constexpr std::uint32_t count = 262144;
        std::vector<float> values(count, 1.0f);
        auto ones = context.create_device_buffer(count * sizeof(float));
        context.upload(ones, values.data(), count * sizeof(float));
        const auto run = [&](std::uint32_t calls, bool fail) {
            auto initial = context.create_device_buffer(count * sizeof(float));
            auto spare = context.create_device_buffer(count * sizeof(float));
            context.upload(initial, values.data(), count * sizeof(float));
            midas_native::VulkanBuffer result;
            bool injected = false;
            try {
                context.batch_external_segmented({}, [&] {
                    auto current = std::move(initial);
                    auto next = std::move(spare);
                    for (std::uint32_t i = 0; i < calls; ++i) {
                        context.dispatch(pipeline, {&next, &current, &ones},
                            &count, sizeof(count), count / 256);
                        std::swap(current, next);
                        if (fail && i == 6) throw std::runtime_error("injected recording failure");
                    }
                    result = std::move(current);
                });
            } catch (const std::runtime_error& error) {
                if (std::string(error.what()) != "injected recording failure") throw;
                injected = true;
            }
            if (injected != fail) throw std::runtime_error("failure injection was not observed");
            if (!fail) {
                std::vector<float> output(count);
                context.download(result, output.data(), count * sizeof(float));
                if (!std::all_of(output.begin(), output.end(), [calls](float v) { return v == calls + 1.0f; }))
                    throw std::runtime_error("cross-segment data or recovery mismatch");
                // Includes the mandatory final command even at an exact boundary.
                if (context.last_external_segment_count() != calls / 4 + 1)
                    throw std::runtime_error("unexpected segment count");
            }
        };
        for (int repeat = 0; repeat < 8; ++repeat) {
            run(12, false);
            run(9, true);
            run(13, false);
        }
        std::cout << "SEGMENTED_BATCH_RECOVERY_OK: exact boundary, cross-segment data, and eight recording-failure recoveries\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
