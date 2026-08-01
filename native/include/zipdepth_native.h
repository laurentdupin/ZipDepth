#ifndef ZIPDEPTH_NATIVE_H
#define ZIPDEPTH_NATIVE_H

#include <stdint.h>

#if defined(_WIN32)
#  if defined(ZIPDEPTH_BUILD_DLL)
#    define ZIPDEPTH_API __declspec(dllexport)
#  else
#    define ZIPDEPTH_API __declspec(dllimport)
#  endif
#  define ZIPDEPTH_CALL __cdecl
#else
#  define ZIPDEPTH_API __attribute__((visibility("default")))
#  define ZIPDEPTH_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define ZIPDEPTH_ABI_VERSION 1u

typedef struct zipdepth_context zipdepth_context;

typedef enum zipdepth_status {
    ZIPDEPTH_STATUS_OK = 0,
    ZIPDEPTH_STATUS_INVALID_ARGUMENT = 1,
    ZIPDEPTH_STATUS_MODEL_IO = 2,
    ZIPDEPTH_STATUS_MODEL_FORMAT = 3,
    ZIPDEPTH_STATUS_VULKAN_UNAVAILABLE = 4,
    ZIPDEPTH_STATUS_OUT_OF_MEMORY = 5,
    ZIPDEPTH_STATUS_INFERENCE_FAILED = 6,
    ZIPDEPTH_STATUS_UNSUPPORTED = 7,
    ZIPDEPTH_STATUS_INTERNAL_ERROR = 8
} zipdepth_status;

typedef enum zipdepth_model_kind {
    ZIPDEPTH_MODEL_BASE_GPU = 0,
    ZIPDEPTH_MODEL_BASE_MOBILE = 1
} zipdepth_model_kind;

typedef struct zipdepth_transfer_counters {
    uint32_t struct_size;
    uint32_t api_version;
    uint64_t tensor_upload_bytes;
    uint64_t tensor_download_bytes;
} zipdepth_transfer_counters;

ZIPDEPTH_API uint32_t ZIPDEPTH_CALL zipdepth_abi_version(void);
ZIPDEPTH_API const char* ZIPDEPTH_CALL zipdepth_last_error(void);
ZIPDEPTH_API zipdepth_status ZIPDEPTH_CALL zipdepth_get_transfer_counters(
    zipdepth_transfer_counters* counters);
ZIPDEPTH_API zipdepth_status ZIPDEPTH_CALL zipdepth_create_vulkan(
    const char* model_path_utf8,
    uint32_t device_index,
    zipdepth_context** out_context);
ZIPDEPTH_API void ZIPDEPTH_CALL zipdepth_destroy(zipdepth_context* context);
ZIPDEPTH_API zipdepth_status ZIPDEPTH_CALL zipdepth_infer_rgb_f32(
    zipdepth_context* context,
    const float* rgb_chw,
    uint32_t width,
    uint32_t height,
    float* depth_hw,
    uint64_t depth_elements);
ZIPDEPTH_API zipdepth_status ZIPDEPTH_CALL zipdepth_infer_tensor_vulkan_f32(
    zipdepth_context* context,
    const float* normalized_rgb_chw,
    uint32_t width,
    uint32_t height,
    float* depth_hw,
    uint64_t depth_elements);

#ifdef __cplusplus
}
#endif

#endif
