#include "inferbridge_harness.h"

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

namespace {

ibrh_string_view view(const std::string& value) {
    return {value.data(), value.size()};
}

bool terminal(std::uint32_t state) {
    return state == IBRH_JOB_COMPLETE || state == IBRH_JOB_FAILED ||
        state == IBRH_JOB_CANCELLED;
}

bool parse_u32(const char* text, std::uint32_t& value) {
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(text, &end, 10);
    if (text == end || *end != '\0' || parsed == 0 || parsed > UINT32_MAX)
        return false;
    value = static_cast<std::uint32_t>(parsed);
    return true;
}

double milliseconds(
    std::chrono::steady_clock::time_point begin,
    std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

int fail(const char* message, ibrh_result result = IBRH_ERROR_INTERNAL) {
    std::fprintf(stderr, "%s (result=%d)\n", message, result);
    return 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 8) {
        std::fprintf(stderr,
            "usage: %s HARNESS_SO MODEL SIZE WIDTH HEIGHT WARMUP ITERATIONS\n",
            argv[0]);
        return 2;
    }
    std::uint32_t size = 0, width = 0, height = 0, warmup = 0, iterations = 0;
    if (!parse_u32(argv[3], size) || !parse_u32(argv[4], width) ||
        !parse_u32(argv[5], height) || !parse_u32(argv[6], warmup) ||
        !parse_u32(argv[7], iterations))
        return fail("invalid numeric argument");

    void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library) return fail(dlerror());
    auto get_api = reinterpret_cast<ibrh_get_api_fn>(
        dlsym(library, "ibrh_get_api"));
    if (!get_api) return fail("ibrh_get_api is not exported");
    ibrh_api api{};
    ibrh_result result = get_api(IBRH_CURRENT_API_VERSION, sizeof(api), &api);
    if (result != IBRH_OK) return fail("ibrh_get_api failed", result);
    ibrh_capabilities capabilities{};
    result = api.query_capabilities(sizeof(capabilities), &capabilities);
    const std::uint64_t required = IBRH_CAP_HOST_MEMORY |
        IBRH_CAP_ASYNC_SUBMIT | IBRH_CAP_CANCELLATION;
    if (result != IBRH_OK || (capabilities.flags & required) != required)
        return fail("required HOST async capabilities are absent", result);

    const std::string backend = "Native";
    const std::string device = R"({"index":0})";
    ibrh_runtime_create_request runtime_request{};
    runtime_request.struct_size = sizeof(runtime_request);
    runtime_request.api_version = IBRH_CURRENT_API_VERSION;
    runtime_request.backend = view(backend);
    runtime_request.requested_device_json = view(device);
    ibrh_runtime* runtime = nullptr;
    result = api.runtime_create(
        sizeof(runtime_request), &runtime_request, &runtime);
    if (result != IBRH_OK) return fail("runtime_create failed", result);

    const std::string model_path = argv[2];
    const std::string parameters =
        std::string("{\"Size\":") + std::to_string(size) + "}";
    ibrh_model_load_request load_request{};
    load_request.struct_size = sizeof(load_request);
    load_request.api_version = IBRH_CURRENT_API_VERSION;
    load_request.model_path = view(model_path);
    load_request.parameters_json = view(parameters);
    ibrh_model* model = nullptr;
    const auto load_begin = std::chrono::steady_clock::now();
    result = api.model_load(
        runtime, sizeof(load_request), &load_request, &model);
    const double load_ms = milliseconds(
        load_begin, std::chrono::steady_clock::now());
    if (result != IBRH_OK) return fail("model_load failed", result);

    const std::uint64_t pixels = static_cast<std::uint64_t>(width) * height;
    std::vector<std::uint8_t> image(static_cast<std::size_t>(pixels) * 4u);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            std::uint8_t* pixel = image.data() +
                (static_cast<std::uint64_t>(y) * width + x) * 4u;
            pixel[0] = static_cast<std::uint8_t>(x * 255u / width);
            pixel[1] = static_cast<std::uint8_t>(y * 255u / height);
            pixel[2] = static_cast<std::uint8_t>((x + y) * 255u /
                (width + height));
            pixel[3] = 255u;
        }
    }
    const auto resource = [width, height](
            std::uint32_t access, std::uint32_t format,
            std::uint32_t stride, std::uint64_t bytes, void* pointer) {
        ibrh_resource value{};
        value.struct_size = sizeof(value);
        value.api_version = IBRH_CURRENT_API_VERSION;
        value.domain = IBRH_RESOURCE_DOMAIN_HOST;
        value.kind = IBRH_RESOURCE_KIND_IMAGE_2D;
        value.access = access;
        value.pixel_format = format;
        value.width = width;
        value.height = height;
        value.depth = 1u;
        value.row_stride_bytes = stride;
        value.native_handle_type = IBRH_NATIVE_HANDLE_HOST_POINTER;
        value.byte_size = bytes;
        value.native_handle = reinterpret_cast<std::uintptr_t>(pointer);
        return value;
    };
    ibrh_transfer_binding input{};
    input.struct_size = sizeof(input);
    input.api_version = IBRH_CURRENT_API_VERSION;
    input.resource = resource(
        IBRH_RESOURCE_ACCESS_READ, IBRH_PIXEL_BGRA8, width * 4u,
        image.size(), image.data());
    ibrh_output_plan_request plan_request{};
    plan_request.struct_size = sizeof(plan_request);
    plan_request.api_version = IBRH_CURRENT_API_VERSION;
    plan_request.inputs = &input.resource;
    plan_request.input_count = 1u;
    plan_request.parameters_json = view(parameters);
    ibrh_port_descriptor planned_output{};
    result = api.model_plan_outputs(
        model, sizeof(plan_request), &plan_request, 1u, &planned_output);
    if (result != IBRH_OK || planned_output.width == 0u ||
        planned_output.height == 0u)
        return fail("model_plan_outputs failed", result);
    const std::uint64_t output_pixels =
        static_cast<std::uint64_t>(planned_output.width) * planned_output.height;
    std::vector<float> depth(static_cast<std::size_t>(output_pixels));
    ibrh_transfer_binding output{};
    output.struct_size = sizeof(output);
    output.api_version = IBRH_CURRENT_API_VERSION;
    output.resource = resource(
        IBRH_RESOURCE_ACCESS_WRITE, IBRH_PIXEL_DEPTH_FLOAT32,
        planned_output.width * sizeof(float), depth.size() * sizeof(float),
        depth.data());
    output.resource.width = planned_output.width;
    output.resource.height = planned_output.height;

    double total_ms = 0.0, minimum_ms = 1.0e30, maximum_ms = 0.0;
    double maximum_submit_ms = 0.0;
    const std::uint32_t calls = warmup + iterations;
    for (std::uint32_t call = 0; call < calls; ++call) {
        ibrh_submit_request submit_request{};
        submit_request.struct_size = sizeof(submit_request);
        submit_request.api_version = IBRH_CURRENT_API_VERSION;
        submit_request.inputs = &input;
        submit_request.input_count = 1u;
        submit_request.outputs = &output;
        submit_request.output_count = 1u;
        submit_request.source_frame_id = call + 1u;
        submit_request.parameters_json = view(parameters);
        ibrh_job* job = nullptr;
        const auto begin = std::chrono::steady_clock::now();
        result = api.submit(model, sizeof(submit_request), &submit_request, &job);
        const auto submitted = std::chrono::steady_clock::now();
        const double submit_ms = milliseconds(begin, submitted);
        maximum_submit_ms = std::max(maximum_submit_ms, submit_ms);
        if (result != IBRH_OK || !job) return fail("submit failed", result);
        ibrh_job_status status{};
        const auto deadline = begin + std::chrono::seconds(30);
        do {
            result = api.job_poll(job, sizeof(status), &status);
            if (result != IBRH_OK) return fail("job_poll failed", result);
            if (!terminal(status.state))
                std::this_thread::sleep_for(std::chrono::microseconds(100));
        } while (!terminal(status.state) &&
                 std::chrono::steady_clock::now() < deadline);
        const auto end = std::chrono::steady_clock::now();
        if (status.state != IBRH_JOB_COMPLETE ||
            status.source_frame_id != call + 1u)
            return fail("job did not complete");
        api.job_release(job);
        if (call >= warmup) {
            const double elapsed = milliseconds(begin, end);
            total_ms += elapsed;
            minimum_ms = std::min(minimum_ms, elapsed);
            maximum_ms = std::max(maximum_ms, elapsed);
        }
    }
    float output_min = depth[0], output_max = depth[0];
    for (float value : depth) {
        if (!std::isfinite(value)) return fail("non-finite depth output");
        output_min = std::min(output_min, value);
        output_max = std::max(output_max, value);
    }
    const double mean_ms = total_ms / iterations;
    std::printf(
        "load_ms=%.3f width=%u height=%u output_width=%u output_height=%u "
        "size=%u warmup=%u iterations=%u "
        "mean_ms=%.3f min_ms=%.3f max_ms=%.3f fps=%.3f "
        "max_submit_ms=%.3f output_min=%.9g output_max=%.9g\n",
        load_ms, width, height, planned_output.width, planned_output.height,
        size, warmup, iterations,
        mean_ms, minimum_ms, maximum_ms, 1000.0 / mean_ms,
        maximum_submit_ms, output_min, output_max);
    api.model_unload(model);
    api.runtime_destroy(runtime);
    if (dlclose(library) != 0) return fail("dlclose failed");
    return 0;
}
