
#include "inferbridge_harness.h"

#include "zipdepth_native.h"
#if defined(ZIPDEPTH_WITH_VULKAN)
#include "external_gpu.h"
#endif

#include <atomic>
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

struct ibrh_runtime {
    std::string error;
    int32_t vulkan_device_index = 0;
    uint64_t adapter_luid = 0u;
};

namespace { class ZipDepthGpuWorker; class ZipDepthHostWorker; }

struct ibrh_model {
    ibrh_runtime* runtime = nullptr;
    zipdepth_context* context = nullptr;
    std::string model_path;
#if defined(ZIPDEPTH_WITH_VULKAN)
    std::shared_ptr<zipdepth_native::ExternalGpu> external_gpu;
    std::shared_ptr<ZipDepthGpuWorker> gpu_worker;
    std::shared_ptr<std::atomic<uint32_t>> gpu_admissions =
        std::make_shared<std::atomic<uint32_t>>(0u);
#endif
    std::shared_ptr<ZipDepthHostWorker> host_worker;
    std::shared_ptr<std::atomic<uint32_t>> host_admissions =
        std::make_shared<std::atomic<uint32_t>>(0u);
    uint32_t input_size = 384u;
    std::mutex submit_mutex;
};

struct ibrh_job {
    std::atomic<uint32_t> references{1u};
    std::atomic<uint32_t> state{IBRH_JOB_QUEUED};
    std::atomic<bool> cancel_requested{false};
#if defined(ZIPDEPTH_WITH_VULKAN)
    mutable std::mutex gpu_mutex;
    std::shared_ptr<zipdepth_native::ExternalJob> gpu_job;
#endif
    uint64_t source_frame_id = 0u;
    uint64_t timestamp_ns = 0u;
    uint32_t width = 0u;
    uint32_t height = 0u;
    bool gpu_backed = false;
    std::shared_ptr<std::atomic<uint32_t>> gpu_admission;
    std::vector<uint8_t> depth;
    ~ibrh_job() {
        if (gpu_admission) gpu_admission->fetch_sub(1u);
    }
};


namespace {

thread_local std::string g_last_error;
constexpr char kHarnessId[] = "inferbridge.zipdepth.native";
constexpr char kHarnessVersion[] = "1.1.0";

ibrh_result fail(
    ibrh_runtime* runtime, ibrh_result result, const std::string& message) {
    g_last_error = message;
    if (runtime != nullptr) runtime->error = message;
    return result;
}
std::string copy_string(ibrh_string_view value) {
    return value.size == 0u ? std::string() :
        std::string(value.data, value.size);
}

bool valid_string(ibrh_string_view value) {
    return value.data != nullptr && value.size != 0u &&
        std::memchr(value.data, '\0', value.size) == nullptr;
}

bool json_string(
    const std::string& json, const std::string& key, std::string& value) {
    const std::string marker = "\"" + key + "\"";
    size_t position = json.find(marker);
    if (position == std::string::npos) return false;
    position = json.find(':', position + marker.size());
    if (position == std::string::npos) return false;
    position = json.find_first_not_of(" \t\r\n", position + 1u);
    if (position == std::string::npos || json[position] != '"') return false;
    const size_t end = json.find('"', position + 1u);
    if (end == std::string::npos) return false;
    value = json.substr(position + 1u, end - position - 1u);
    return true;
}

bool json_uint(
    const std::string& json, const std::string& key, uint32_t& value) {
    const std::string marker = "\"" + key + "\"";
    size_t position = json.find(marker);
    if (position == std::string::npos) return false;
    position = json.find(':', position + marker.size());
    if (position == std::string::npos) return false;
    position = json.find_first_not_of(" \t\r\n", position + 1u);
    if (position == std::string::npos) return false;
    if (json[position] == '"') ++position;
    size_t end = position;
    while (end < json.size() && json[end] >= '0' && json[end] <= '9') ++end;
    if (end == position) return false;
    uint64_t parsed = 0u;
    for (size_t index = position; index < end; ++index) {
        parsed = parsed * 10u + static_cast<uint32_t>(json[index] - '0');
        if (parsed > std::numeric_limits<uint32_t>::max()) return false;
    }
    value = static_cast<uint32_t>(parsed);
    return true;
}

bool parse_luid(const std::string& value, uint64_t& result) {
    if (value.size() != 16u) return false;
    const auto nibble = [](char character) -> int {
        if (character >= '0' && character <= '9') return character - '0';
        if (character >= 'a' && character <= 'f') return character - 'a' + 10;
        if (character >= 'A' && character <= 'F') return character - 'A' + 10;
        return -1;
    };
    uint8_t bytes[8]{};
    for (size_t index = 0; index < 8u; ++index) {
        const int high = nibble(value[index * 2u]);
        const int low = nibble(value[index * 2u + 1u]);
        if (high < 0 || low < 0) return false;
        bytes[index] = static_cast<uint8_t>((high << 4) | low);
    }
    std::memcpy(&result, bytes, sizeof(result));
    return true;
}

bool device_index_for_luid(uint64_t luid, int32_t& device_index) {
#if defined(ZIPDEPTH_WITH_VULKAN) && defined(_WIN32)
    for (int32_t index = 0; index < 32; ++index) {
        try {
            const auto capabilities =
                zipdepth_native::probe_external_gpu(static_cast<uint32_t>(index));
            if (capabilities.available && capabilities.adapter_luid == luid) {
                device_index = index;
                return true;
            }
        } catch (...) {
            if (index == 0) return false;
            break;
        }
    }
#else
    (void)luid;
    (void)device_index;
#endif
    return false;
}

bool input_size(
    const std::string& json, uint32_t fallback, uint32_t& value) {
    value = fallback;
    uint32_t parsed = 0u;
    if (!json_uint(json, "Size", parsed)) return true;
    if (parsed == 0u || parsed > 4096u) return false;
    value = parsed;
    return true;
}

ibrh_result status_result(zipdepth_status status) {
    switch (status) {
        case ZIPDEPTH_STATUS_OK: return IBRH_OK;
        case ZIPDEPTH_STATUS_INVALID_ARGUMENT:
            return IBRH_ERROR_INVALID_ARGUMENT;
        case ZIPDEPTH_STATUS_VULKAN_UNAVAILABLE:
        case ZIPDEPTH_STATUS_UNSUPPORTED:
            return IBRH_ERROR_UNSUPPORTED_CAPABILITY;
        default:
            return IBRH_ERROR_INTERNAL;
    }
}

void retain_job(ibrh_job* job) {
    (void)job->references.fetch_add(1u);
}

void release_job(ibrh_job* job) {
    if (job != nullptr && job->references.fetch_sub(1u) == 1u) delete job;
}

class ZipDepthHostWorker final {
public:
    struct Work {
        ibrh_job* job = nullptr;
        zipdepth_context* context = nullptr;
        const uint8_t* pixels = nullptr;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t row_stride = 0;
        bool rgba = false;
        uint32_t network_size = 0;
        float* destination = nullptr;
        uint32_t destination_stride = 0;
    };

    ZipDepthHostWorker() : thread_([this] { run(); }) {}
    ~ZipDepthHostWorker() {
        std::deque<Work> dropped;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            dropped.swap(queue_);
        }
        for (Work& work : dropped) {
            work.job->state.store(IBRH_JOB_CANCELLED);
            release_job(work.job);
        }
        condition_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

    bool enqueue(Work work) {
        retain_job(work.job);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) {
                release_job(work.job);
                return false;
            }
            queue_.push_back(work);
        }
        condition_.notify_one();
        return true;
    }

private:
    static uint32_t aligned_size(double value) {
        return std::max(32u, static_cast<uint32_t>(
            std::llround(value / 32.0) * 32.0));
    }

    void execute(const Work& work) {
        const double scale = static_cast<double>(work.network_size) /
            static_cast<double>(std::min(work.width, work.height));
        const uint32_t network_width = aligned_size(work.width * scale);
        const uint32_t network_height = aligned_size(work.height * scale);
        const uint64_t plane = static_cast<uint64_t>(network_width) *
            network_height;
        std::vector<float> rgb(static_cast<size_t>(plane) * 3u);
        for (uint32_t y = 0; y < network_height; ++y) {
            const uint32_t source_y = std::min(
                work.height - 1u,
                static_cast<uint32_t>(
                    static_cast<uint64_t>(y) * work.height / network_height));
            for (uint32_t x = 0; x < network_width; ++x) {
                const uint32_t source_x = std::min(
                    work.width - 1u,
                    static_cast<uint32_t>(
                        static_cast<uint64_t>(x) * work.width / network_width));
                const uint8_t* pixel = work.pixels +
                    static_cast<uint64_t>(source_y) * work.row_stride +
                    static_cast<uint64_t>(source_x) * 4u;
                const uint64_t index =
                    static_cast<uint64_t>(y) * network_width + x;
                rgb[index] = pixel[work.rgba ? 0u : 2u] / 255.0f;
                rgb[plane + index] = pixel[1] / 255.0f;
                rgb[2u * plane + index] =
                    pixel[work.rgba ? 2u : 0u] / 255.0f;
            }
        }
        std::vector<float> temporary(static_cast<size_t>(plane));
        const zipdepth_status result = zipdepth_infer_tensor_vulkan_f32(
            work.context, rgb.data(), network_width, network_height,
            temporary.data(), temporary.size());
        if (result != ZIPDEPTH_STATUS_OK)
            throw std::runtime_error(zipdepth_last_error());
        for (uint32_t y = 0; y < work.height; ++y) {
            const uint32_t source_y = y * network_height / work.height;
            for (uint32_t x = 0; x < work.width; ++x) {
                const uint32_t source_x = x * network_width / work.width;
                work.destination[
                    static_cast<uint64_t>(y) * work.destination_stride + x] =
                    temporary[static_cast<uint64_t>(source_y) *
                        network_width + source_x];
            }
        }
    }

    void run() {
        for (;;) {
            Work work;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                condition_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
                if (stopping_ && queue_.empty()) return;
                work = queue_.front();
                queue_.pop_front();
            }
            uint32_t queued = IBRH_JOB_QUEUED;
            if (work.job->cancel_requested.load()) {
                work.job->state.store(IBRH_JOB_CANCELLED);
            } else if (work.job->state.compare_exchange_strong(
                    queued, IBRH_JOB_RUNNING)) {
                try {
                    execute(work);
                    uint32_t running = IBRH_JOB_RUNNING;
                    work.job->state.compare_exchange_strong(running,
                        work.job->cancel_requested.load() ?
                            IBRH_JOB_CANCELLED : IBRH_JOB_COMPLETE);
                } catch (...) {
                    work.job->state.store(IBRH_JOB_FAILED);
                }
            }
            release_job(work.job);
        }
    }

    std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<Work> queue_;
    bool stopping_ = false;
    std::thread thread_;
};

#if defined(ZIPDEPTH_WITH_VULKAN)
class ZipDepthGpuWorker final {
public:
    explicit ZipDepthGpuWorker(
        std::shared_ptr<zipdepth_native::ExternalGpu> external_gpu)
        : external_gpu_(std::move(external_gpu)),
          thread_([this] { run(); }) {}

    ~ZipDepthGpuWorker() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            for (Work& work : queue_) {
                work.job->state.store(IBRH_JOB_CANCELLED);
                release_job(work.job);
            }
            queue_.clear();
        }
        condition_.notify_one();
        if (thread_.joinable()) thread_.join();
    }

    bool enqueue(
        ibrh_job* job,
        const zipdepth_native::ExternalTextureRequest& request) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_) return false;
        if (queue_.size() >= 3u) {
            Work dropped = queue_.front();
            queue_.pop_front();
            dropped.job->state.store(IBRH_JOB_CANCELLED);
            release_job(dropped.job);
        }
        retain_job(job);
        queue_.push_back({job, request});
        condition_.notify_one();
        return true;
    }

private:
    struct Work {
        ibrh_job* job;
        zipdepth_native::ExternalTextureRequest request;
    };

    void run() {
        for (;;) {
            Work work{};
            {
                std::unique_lock<std::mutex> lock(mutex_);
                condition_.wait(lock, [this] {
                    return stopping_ || !queue_.empty();
                });
                if (stopping_ && queue_.empty()) return;
                work = queue_.front();
                queue_.pop_front();
            }
            uint32_t queued = IBRH_JOB_QUEUED;
            if (work.job->state.compare_exchange_strong(
                    queued, IBRH_JOB_RUNNING)) {
                try {
                    auto gpu_job = external_gpu_->submit_texture(work.request);
                    std::lock_guard<std::mutex> lock(work.job->gpu_mutex);
                    work.job->gpu_job = std::move(gpu_job);
                    if (work.job->state.load() == IBRH_JOB_CANCELLED)
                        work.job->gpu_job->cancel();
                } catch (...) {
                    work.job->state.store(IBRH_JOB_FAILED);
                }
            }
            release_job(work.job);
        }
    }

    std::shared_ptr<zipdepth_native::ExternalGpu> external_gpu_;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<Work> queue_;
    bool stopping_ = false;
    std::thread thread_;
};
#endif

ibrh_result IBRH_CALL query_capabilities(
    size_t capabilities_size, ibrh_capabilities* capabilities) {
    if (capabilities == nullptr) return IBRH_ERROR_INVALID_ARGUMENT;
    if (capabilities_size < sizeof(*capabilities))
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    *capabilities = {};
    capabilities->struct_size = sizeof(*capabilities);
    capabilities->api_version = IBRH_CURRENT_API_VERSION;
    capabilities->flags = IBRH_CAP_HOST_MEMORY |
        IBRH_CAP_ASYNC_SUBMIT | IBRH_CAP_CANCELLATION;
    capabilities->input_domain_mask =
        1ull << IBRH_RESOURCE_DOMAIN_HOST;
    capabilities->output_domain_mask =
        1ull << IBRH_RESOURCE_DOMAIN_HOST;
    capabilities->maximum_inputs = 1u;
    capabilities->maximum_outputs = 1u;
    capabilities->maximum_in_flight_jobs = 3u;
#if defined(ZIPDEPTH_WITH_VULKAN) && defined(_WIN32)
    try {
        if (zipdepth_native::probe_external_gpu(0u).available) {
            capabilities->flags |=
                IBRH_CAP_ASYNC_SUBMIT | IBRH_CAP_CANCELLATION |
                IBRH_CAP_GPU_RESOURCES | IBRH_CAP_EXTERNAL_SYNCHRONIZATION |
                IBRH_CAP_GPU_RESIDENT_OUTPUT;
            capabilities->input_domain_mask |=
                1ull << IBRH_RESOURCE_DOMAIN_D3D12;
            capabilities->output_domain_mask |=
                1ull << IBRH_RESOURCE_DOMAIN_D3D12;
            capabilities->synchronization_mask =
                1ull << IBRH_SYNC_D3D12_FENCE;
            capabilities->maximum_in_flight_jobs = 3u;
        }
    } catch (...) {
    }
#endif
    capabilities->harness_id = {kHarnessId, sizeof(kHarnessId) - 1u};
    capabilities->harness_version = {
        kHarnessVersion, sizeof(kHarnessVersion) - 1u};
    return IBRH_OK;
}

ibrh_result IBRH_CALL runtime_create(
    size_t request_size, const ibrh_runtime_create_request* request,
    ibrh_runtime** output) {
    if (request == nullptr || output == nullptr)
        return IBRH_ERROR_INVALID_ARGUMENT;
    *output = nullptr;
    if (request_size < sizeof(*request) ||
        request->struct_size < sizeof(*request))
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    auto* runtime = new (std::nothrow) ibrh_runtime();
    if (runtime == nullptr) return IBRH_ERROR_INTERNAL;
    const std::string device = copy_string(request->requested_device_json);
    uint32_t index = 0u;
    if (json_uint(device, "index", index)) {
        if (index > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
            delete runtime;
            return fail(
                nullptr, IBRH_ERROR_INVALID_ARGUMENT,
                "ZipDepth requested device index is out of range");
        }
        runtime->vulkan_device_index = static_cast<int32_t>(index);
    }
    std::string luid_text;
    if (json_string(device, "luid", luid_text) && !luid_text.empty()) {
        uint64_t luid = 0u;
        if (!parse_luid(luid_text, luid) ||
            !device_index_for_luid(luid, runtime->vulkan_device_index)) {
            delete runtime;
            return fail(
                nullptr, IBRH_ERROR_UNSUPPORTED_CAPABILITY,
                "ZipDepth could not match the requested GPU LUID");
        }
        runtime->adapter_luid = luid;
    }
    *output = runtime;
    return IBRH_OK;
}

void IBRH_CALL runtime_destroy(ibrh_runtime* runtime) {
    delete runtime;
}

ibrh_result IBRH_CALL model_load(
    ibrh_runtime* runtime, size_t request_size,
    const ibrh_model_load_request* request, ibrh_model** output) {
    if (runtime == nullptr || request == nullptr || output == nullptr)
        return IBRH_ERROR_INVALID_ARGUMENT;
    *output = nullptr;
    if (request_size < sizeof(*request) ||
        request->struct_size < sizeof(*request))
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    if (!valid_string(request->model_path))
        return fail(
            runtime, IBRH_ERROR_INVALID_ARGUMENT,
            "ZipDepth model path is missing");
    const std::string path = copy_string(request->model_path);
    const std::string parameters = copy_string(request->parameters_json);
    std::string weights;
    (void)json_string(parameters, "Weights", weights);
    auto* model = new (std::nothrow) ibrh_model();
    if (model == nullptr) return IBRH_ERROR_INTERNAL;
    model->runtime = runtime;
    model->model_path = path;
    if (!input_size(parameters, model->input_size, model->input_size)) {
        delete model;
        return fail(
            runtime, IBRH_ERROR_INVALID_ARGUMENT,
            "ZipDepth Size must be an integer from 1 to 4096");
    }
#if defined(ZIPDEPTH_WITH_VULKAN) && defined(_WIN32)
    if (runtime->adapter_luid != 0u) {
        try {
            model->external_gpu = zipdepth_native::create_external_gpu(
                path, static_cast<uint32_t>(runtime->vulkan_device_index));
            const auto capabilities = model->external_gpu->capabilities();
            if (!capabilities.available ||
                capabilities.adapter_luid != runtime->adapter_luid) {
                throw std::runtime_error(
                    "ZipDepth loaded on a GPU other than the requested LUID");
            }
            model->gpu_worker = std::make_shared<ZipDepthGpuWorker>(
                model->external_gpu);
        } catch (const std::exception& error) {
            delete model;
            return fail(
                runtime, IBRH_ERROR_UNSUPPORTED_CAPABILITY, error.what());
        }
    } else
#endif
    {
        const zipdepth_status status = zipdepth_create_vulkan(
            path.c_str(), static_cast<uint32_t>(runtime->vulkan_device_index),
            &model->context);
        if (status != ZIPDEPTH_STATUS_OK) {
            const std::string message =
                std::string("ZipDepth model load failed: ") + zipdepth_last_error();
            delete model;
            return fail(runtime, status_result(status), message);
        }
        model->host_worker = std::make_shared<ZipDepthHostWorker>();
    }
    *output = model;
    return IBRH_OK;
}

void IBRH_CALL model_unload(ibrh_model* model) {
    if (model == nullptr) return;
    model->host_worker.reset();
#if defined(ZIPDEPTH_WITH_VULKAN)
    model->gpu_worker.reset();
    model->external_gpu.reset();
#endif
    zipdepth_destroy(model->context);
    delete model;
}

ibrh_result IBRH_CALL model_describe_io(
    const ibrh_model* model, size_t size, ibrh_model_io_descriptor* out) {
    if (!model || !out) return IBRH_ERROR_INVALID_ARGUMENT;
    if (size < sizeof(*out)) return IBRH_ERROR_STRUCT_TOO_SMALL;
    *out = {}; out->struct_size = sizeof(*out);
    out->api_version = IBRH_CURRENT_API_VERSION;
    out->input_count = 1u; out->output_count = 1u;
    return IBRH_OK;
}
ibrh_result IBRH_CALL model_get_port(
    const ibrh_model* model, uint32_t direction, uint32_t index,
    size_t size, ibrh_port_descriptor* out) {
    if (!model || !out) return IBRH_ERROR_INVALID_ARGUMENT;
    if (size < sizeof(*out)) return IBRH_ERROR_STRUCT_TOO_SMALL;
    if (index || (direction != IBRH_PORT_INPUT && direction != IBRH_PORT_OUTPUT))
        return IBRH_ERROR_NOT_FOUND;
    *out = {}; out->struct_size = sizeof(*out);
    out->api_version = IBRH_CURRENT_API_VERSION; out->index = 0u;
    out->direction = direction;
    out->semantic = direction == IBRH_PORT_INPUT ? IBRH_SEMANTIC_IMAGE : IBRH_SEMANTIC_DEPTH;
    out->payload_type = direction == IBRH_PORT_INPUT ? IBRH_PIXEL_BGRA8 : IBRH_PIXEL_DEPTH_FLOAT32;
    out->pixel_format = out->payload_type;
    out->accepted_pixel_format_mask = direction == IBRH_PORT_INPUT ?
        ((1ull << IBRH_PIXEL_BGRA8) | (1ull << IBRH_PIXEL_RGBA8)) :
        (1ull << IBRH_PIXEL_DEPTH_FLOAT32);
    out->resource_kind = IBRH_RESOURCE_KIND_IMAGE_2D; out->depth = 1u;
    out->flags = IBRH_DESCRIPTOR_DYNAMIC_WIDTH | IBRH_DESCRIPTOR_DYNAMIC_HEIGHT;
    return IBRH_OK;
}
ibrh_result IBRH_CALL model_plan_outputs(
    const ibrh_model* model, size_t size, const ibrh_output_plan_request* request,
    uint32_t capacity, ibrh_port_descriptor* outputs) {
    if (!model || !request || !outputs) return IBRH_ERROR_INVALID_ARGUMENT;
    if (size < sizeof(*request) || request->struct_size < sizeof(*request))
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    if (capacity < 1u) return IBRH_ERROR_STRUCT_TOO_SMALL;
    if (request->input_count != 1u || !request->inputs ||
        !request->inputs[0].width || !request->inputs[0].height)
        return IBRH_ERROR_INVALID_ARGUMENT;
    const auto result = model_get_port(model, IBRH_PORT_OUTPUT, 0u,
                                       sizeof(outputs[0]), &outputs[0]);
    if (result != IBRH_OK) return result;
    outputs[0].width = request->inputs[0].width;
    outputs[0].height = request->inputs[0].height;
    outputs[0].flags = 0u;
    return IBRH_OK;
}

ibrh_result IBRH_CALL submit(
    ibrh_model* model, size_t request_size,
    const ibrh_submit_request* request, ibrh_job** output) {
    if (!model || !request || !output) return IBRH_ERROR_INVALID_ARGUMENT;
    *output = nullptr;
    if (request_size < sizeof(*request) || request->struct_size < sizeof(*request))
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    if (request->input_count != 1u || !request->inputs ||
        request->output_count != 1u || !request->outputs)
        return IBRH_ERROR_INVALID_ARGUMENT;
    const auto& source = request->inputs[0];
    const auto& target = request->outputs[0];
    if (source.struct_size < sizeof(source) || target.struct_size < sizeof(target))
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    const auto& input = source.resource;
    const auto& destination = target.resource;
    uint32_t network_size = model->input_size;
    if (!input_size(copy_string(request->parameters_json), network_size, network_size))
        return fail(model->runtime, IBRH_ERROR_INVALID_ARGUMENT,
                    "ZipDepth Size must be an integer from 1 to 4096");
    if (!input.width || !input.height || destination.width != input.width ||
        destination.height != input.height ||
        destination.pixel_format != IBRH_PIXEL_DEPTH_FLOAT32)
        return IBRH_ERROR_INVALID_ARGUMENT;
    const uint64_t input_bytes =
        static_cast<uint64_t>(input.row_stride_bytes) * input.height;
    const uint64_t output_bytes =
        static_cast<uint64_t>(destination.row_stride_bytes) *
            destination.height;
    if (input.native_handle == 0u || destination.native_handle == 0u ||
        static_cast<uint64_t>(input.row_stride_bytes) <
            static_cast<uint64_t>(input.width) * 4u ||
        static_cast<uint64_t>(destination.row_stride_bytes) <
            static_cast<uint64_t>(destination.width) * sizeof(float) ||
        destination.row_stride_bytes % sizeof(float) != 0u ||
        input.byte_offset > input.byte_size ||
        input_bytes > input.byte_size - input.byte_offset ||
        destination.byte_offset > destination.byte_size ||
        output_bytes > destination.byte_size - destination.byte_offset)
        return IBRH_ERROR_INVALID_ARGUMENT;
#if defined(ZIPDEPTH_WITH_VULKAN) && defined(_WIN32)
    if (input.domain == IBRH_RESOURCE_DOMAIN_D3D12) {
        if (!model->external_gpu || destination.domain != IBRH_RESOURCE_DOMAIN_D3D12 ||
            input.native_handle_type != IBRH_NATIVE_HANDLE_WIN32_SHARED ||
            destination.native_handle_type != IBRH_NATIVE_HANDLE_WIN32_SHARED ||
            (input.pixel_format != IBRH_PIXEL_BGRA8 && input.pixel_format != IBRH_PIXEL_RGBA8) ||
            source.synchronization.kind != IBRH_SYNC_D3D12_FENCE ||
            source.synchronization.operation != IBRH_SYNC_WAIT ||
            target.synchronization.kind != IBRH_SYNC_D3D12_FENCE ||
            target.synchronization.operation != IBRH_SYNC_SIGNAL)
            return IBRH_ERROR_UNSUPPORTED_CAPABILITY;
        auto* job = new (std::nothrow) ibrh_job();
        if (!job) return IBRH_ERROR_INTERNAL;
        uint32_t admitted = model->gpu_admissions->load();
        while (admitted < 3u && !model->gpu_admissions->compare_exchange_weak(
                admitted, admitted + 1u)) {}
        if (admitted >= 3u) {
            delete job;
            return IBRH_ERROR_INVALID_STATE;
        }
        job->gpu_admission = model->gpu_admissions;
        const zipdepth_native::ExternalTextureRequest gpu_request{
                static_cast<uintptr_t>(input.native_handle), input.width, input.height,
                input.pixel_format == IBRH_PIXEL_RGBA8, network_size,
                static_cast<uintptr_t>(source.synchronization.native_handle),
                source.synchronization.value,
                static_cast<uintptr_t>(destination.native_handle),
                destination.width, destination.height,
                static_cast<uintptr_t>(target.synchronization.native_handle),
                target.synchronization.value,
                request->source_frame_id, request->timestamp_ns};
        job->source_frame_id = request->source_frame_id;
        job->timestamp_ns = request->timestamp_ns;
        job->width = input.width; job->height = input.height;
        job->gpu_backed = true;
        if (!model->gpu_worker ||
            !model->gpu_worker->enqueue(job, gpu_request)) {
            delete job;
            return IBRH_ERROR_INVALID_STATE;
        }
        *output = job; return IBRH_OK;
    }
#endif
    if (input.domain != IBRH_RESOURCE_DOMAIN_HOST ||
        destination.domain != IBRH_RESOURCE_DOMAIN_HOST ||
        input.native_handle_type != IBRH_NATIVE_HANDLE_HOST_POINTER ||
        destination.native_handle_type != IBRH_NATIVE_HANDLE_HOST_POINTER ||
        (input.pixel_format != IBRH_PIXEL_BGRA8 &&
         input.pixel_format != IBRH_PIXEL_RGBA8) ||
        source.synchronization.kind != IBRH_SYNC_NONE ||
        target.synchronization.kind != IBRH_SYNC_NONE)
        return IBRH_ERROR_UNSUPPORTED_CAPABILITY;
    const auto* pixels = reinterpret_cast<const uint8_t*>(
        static_cast<uintptr_t>(input.native_handle)) + input.byte_offset;
    auto* depth = reinterpret_cast<float*>(
        static_cast<uintptr_t>(destination.native_handle) + destination.byte_offset);
    auto* job = new (std::nothrow) ibrh_job();
    if (!job) return IBRH_ERROR_INTERNAL;
    uint32_t admitted = model->host_admissions->load();
    while (admitted < 3u && !model->host_admissions->compare_exchange_weak(
            admitted, admitted + 1u)) {}
    if (admitted >= 3u) {
        delete job;
        return IBRH_ERROR_INVALID_STATE;
    }
    job->gpu_admission = model->host_admissions;
    job->source_frame_id = request->source_frame_id;
    job->timestamp_ns = request->timestamp_ns;
    job->width = input.width; job->height = input.height;
    if (!model->host_worker || !model->host_worker->enqueue({
            job, model->context, pixels, input.width, input.height,
            input.row_stride_bytes, input.pixel_format == IBRH_PIXEL_RGBA8,
            network_size, depth,
            static_cast<uint32_t>(
                destination.row_stride_bytes / sizeof(float))})) {
        delete job;
        return IBRH_ERROR_INVALID_STATE;
    }
    *output = job;
    return IBRH_OK;
}

ibrh_result IBRH_CALL job_poll(
    const ibrh_job* job, size_t status_size, ibrh_job_status* status) {
    if (job == nullptr || status == nullptr)
        return IBRH_ERROR_INVALID_ARGUMENT;
    if (status_size < sizeof(*status)) return IBRH_ERROR_STRUCT_TOO_SMALL;
    *status = {};
    status->struct_size = sizeof(*status);
#if defined(ZIPDEPTH_WITH_VULKAN)
    if (!job->gpu_backed) {
        status->state = job->state.load();
    } else if (job->state.load() == IBRH_JOB_FAILED ||
        job->state.load() == IBRH_JOB_CANCELLED) {
        status->state = job->state.load();
    } else {
        std::lock_guard<std::mutex> lock(job->gpu_mutex);
        if (!job->gpu_job) {
            status->state = job->state.load();
        } else switch (job->gpu_job->state()) {
            case zipdepth_native::ExternalJobState::running:
                status->state = IBRH_JOB_RUNNING;
                break;
            case zipdepth_native::ExternalJobState::complete:
                status->state = IBRH_JOB_COMPLETE;
                break;
            case zipdepth_native::ExternalJobState::cancelled:
                status->state = IBRH_JOB_CANCELLED;
                break;
        }
    }
#endif
    status->output_count = 1u;
    status->source_frame_id = job->source_frame_id;
    return IBRH_OK;
}

ibrh_result IBRH_CALL job_cancel(ibrh_job* job) {
    if (job == nullptr) return IBRH_ERROR_INVALID_ARGUMENT;
#if defined(ZIPDEPTH_WITH_VULKAN)
    if (job->gpu_backed) {
        job->state.store(IBRH_JOB_CANCELLED);
        std::lock_guard<std::mutex> lock(job->gpu_mutex);
        if (job->gpu_job) job->gpu_job->cancel();
    } else {
        job->cancel_requested.store(true);
    }
    return IBRH_OK;
#endif
    return IBRH_ERROR_INVALID_STATE;
}

void IBRH_CALL job_release(ibrh_job* job) {
    release_job(job);
}

ibrh_result IBRH_CALL get_last_error(
    const void* object, char* destination, size_t destination_size,
    size_t* required_size) {
    const auto* runtime = static_cast<const ibrh_runtime*>(object);
    const std::string& message =
        runtime != nullptr && !runtime->error.empty() ?
        runtime->error : g_last_error;
    const size_t required = message.size() + 1u;
    if (required_size != nullptr) *required_size = required;
    if (destination == nullptr || destination_size < required)
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    std::memcpy(destination, message.c_str(), required);
    return IBRH_OK;
}

}  // namespace

extern "C" IBRH_API ibrh_result IBRH_CALL ibrh_get_api(
    uint32_t requested_api_version, size_t api_size, ibrh_api* api) {
    if (api == nullptr) return IBRH_ERROR_INVALID_ARGUMENT;
    if (api_size < sizeof(*api)) return IBRH_ERROR_STRUCT_TOO_SMALL;
    if ((requested_api_version >> 16u) != IBRH_API_VERSION_MAJOR)
        return IBRH_ERROR_UNSUPPORTED_API;
    *api = {};
    api->struct_size = sizeof(*api);
    api->api_version = IBRH_CURRENT_API_VERSION;
    api->query_capabilities = query_capabilities;
    api->runtime_create = runtime_create;
    api->runtime_destroy = runtime_destroy;
    api->model_load = model_load;
    api->model_unload = model_unload;
    api->model_describe_io = model_describe_io;
    api->model_get_port = model_get_port;
    api->model_plan_outputs = model_plan_outputs;
    api->submit = submit;
    api->job_poll = job_poll;
    api->job_cancel = job_cancel;
    api->job_release = job_release;
    api->get_last_error = get_last_error;
    return IBRH_OK;
}
