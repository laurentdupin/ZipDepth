#include "zipdepth_native.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

static double now_ms(void) {
#if defined(_WIN32)
    LARGE_INTEGER counter, frequency;
    QueryPerformanceCounter(&counter);
    QueryPerformanceFrequency(&frequency);
    return (double)counter.QuadPart * 1000.0 / (double)frequency.QuadPart;
#else
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec * 1000.0 + (double)value.tv_nsec / 1000000.0;
#endif
}

static int parse_u32(const char* text, uint32_t* value) {
    char* end = NULL;
    const unsigned long parsed = strtoul(text, &end, 10);
    if (text == end || *end != '\0' || parsed == 0 || parsed > UINT32_MAX)
        return 0;
    *value = (uint32_t)parsed;
    return 1;
}

int main(int argc, char** argv) {
    if (argc != 6) {
        fprintf(stderr,
            "usage: %s MODEL WIDTH HEIGHT WARMUP ITERATIONS\n", argv[0]);
        return 2;
    }
    uint32_t width = 0, height = 0, warmup = 0, iterations = 0;
    if (!parse_u32(argv[2], &width) || !parse_u32(argv[3], &height) ||
        !parse_u32(argv[4], &warmup) || !parse_u32(argv[5], &iterations) ||
        (width % 32u) != 0u || (height % 32u) != 0u) {
        fprintf(stderr, "dimensions must be positive multiples of 32\n");
        return 2;
    }
    const uint64_t pixels = (uint64_t)width * height;
    if (pixels > SIZE_MAX / (3u * sizeof(float))) return 2;
    float* input = (float*)malloc((size_t)pixels * 3u * sizeof(float));
    float* output = (float*)malloc((size_t)pixels * sizeof(float));
    if (!input || !output) return 3;
    for (uint64_t i = 0; i < pixels * 3u; ++i)
        input[i] = (float)((i * 1103515245u + 12345u) & 1023u) / 1023.0f;

    zipdepth_context* context = NULL;
    const double load_begin = now_ms();
    zipdepth_status status = zipdepth_create_vulkan(argv[1], 0u, &context);
    const double load_ms = now_ms() - load_begin;
    if (status != ZIPDEPTH_STATUS_OK) {
        fprintf(stderr, "load failed: %s\n", zipdepth_last_error());
        return 4;
    }
    for (uint32_t i = 0; i < warmup; ++i) {
        status = zipdepth_infer_tensor_vulkan_f32(
            context, input, width, height, output, pixels);
        if (status != ZIPDEPTH_STATUS_OK) {
            fprintf(stderr, "warmup failed: %s\n", zipdepth_last_error());
            return 5;
        }
    }
    double total_ms = 0.0;
    double minimum_ms = 1.0e30;
    double maximum_ms = 0.0;
    for (uint32_t i = 0; i < iterations; ++i) {
        const double begin = now_ms();
        status = zipdepth_infer_tensor_vulkan_f32(
            context, input, width, height, output, pixels);
        const double elapsed = now_ms() - begin;
        if (status != ZIPDEPTH_STATUS_OK) {
            fprintf(stderr, "inference failed: %s\n", zipdepth_last_error());
            return 6;
        }
        total_ms += elapsed;
        if (elapsed < minimum_ms) minimum_ms = elapsed;
        if (elapsed > maximum_ms) maximum_ms = elapsed;
    }
    float output_min = INFINITY, output_max = -INFINITY;
    double output_sum = 0.0;
    for (uint64_t i = 0; i < pixels; ++i) {
        if (output[i] < output_min) output_min = output[i];
        if (output[i] > output_max) output_max = output[i];
        output_sum += output[i];
    }
    const double mean_ms = total_ms / iterations;
    printf("load_ms=%.3f width=%u height=%u warmup=%u iterations=%u "
           "mean_ms=%.3f min_ms=%.3f max_ms=%.3f fps=%.3f "
           "output_min=%.9g output_max=%.9g output_mean=%.9g\n",
        load_ms, width, height, warmup, iterations,
        mean_ms, minimum_ms, maximum_ms, 1000.0 / mean_ms,
        output_min, output_max, output_sum / (double)pixels);
    zipdepth_destroy(context);
    free(output);
    free(input);
    return 0;
}
