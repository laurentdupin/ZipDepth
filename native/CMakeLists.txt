cmake_minimum_required(VERSION 3.20)
project(zipdepth_native VERSION 0.1.0 LANGUAGES C CXX)
include("${CMAKE_CURRENT_LIST_DIR}/../../native_support/InferBridgeNativeHarnessSupport.cmake")
link_libraries(InferBridge::NativeHarnessSupport)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ZIPDEPTH_WITH_VULKAN "Build the Vulkan executor" ON)
option(ZIPDEPTH_WITH_METAL "Build the Apple Metal/MPSGraph executor" ${APPLE})

if(ZIPDEPTH_WITH_VULKAN)
    find_package(Vulkan REQUIRED COMPONENTS glslc)
    set(ZIPDEPTH_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
    file(MAKE_DIRECTORY "${ZIPDEPTH_GENERATED_DIR}")
    function(zipdepth_compile_shader NAME)
        set(SOURCE_NAME "${NAME}")
        set(EXTRA_ARGUMENTS)
        if(ARGC GREATER 1)
            set(SOURCE_NAME "${ARGV1}")
        endif()
        if(ARGC GREATER 2)
            set(EXTRA_ARGUMENTS ${ARGN})
            list(POP_FRONT EXTRA_ARGUMENTS)
        endif()
        set(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/${SOURCE_NAME}.glsl")
        set(SPIRV "${ZIPDEPTH_GENERATED_DIR}/${NAME}.spv")
        set(HEADER "${ZIPDEPTH_GENERATED_DIR}/${NAME}_spv.h")
        add_custom_command(OUTPUT "${SPIRV}"
            COMMAND Vulkan::glslc -O -fshader-stage=compute
                --target-env=vulkan1.3 --target-spv=spv1.6 -Werror
                ${EXTRA_ARGUMENTS} -o "${SPIRV}" "${SOURCE}"
            DEPENDS "${SOURCE}" VERBATIM)
        add_custom_command(OUTPUT "${HEADER}"
            COMMAND "${CMAKE_COMMAND}" -DINPUT=${SPIRV} -DOUTPUT=${HEADER}
                -DVARIABLE=midas_${NAME}_spv
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake"
            DEPENDS "${SPIRV}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake"
            VERBATIM)
        set(ZIPDEPTH_SHADER_HEADERS ${ZIPDEPTH_SHADER_HEADERS} "${HEADER}" PARENT_SCOPE)
    endfunction()
    zipdepth_compile_shader(conv2d_grouped conv2d_grouped -DNO_BATCH_NORM=1)
    zipdepth_compile_shader(conv2d_pointwise4 conv2d_pointwise4 -DNO_BATCH_NORM=1)
    zipdepth_compile_shader(conv2d_pointwise_gemm conv2d_pointwise_gemm -DNO_BATCH_NORM=1)
    zipdepth_compile_shader(conv2d_pointwise_gemm_residual conv2d_pointwise_gemm -DADD_RESIDUAL=1)
    zipdepth_compile_shader(conv2d_depthwise3 conv2d_depthwise3 -DNO_BATCH_NORM=1)
    zipdepth_compile_shader(conv2d_spatial4 conv2d_spatial4 -DNO_BATCH_NORM=1)
    zipdepth_compile_shader(conv2d_spatial4_tiled conv2d_spatial4_tiled
        -DNO_BATCH_NORM=1 -DLOCAL_X=16 -DLOCAL_Y=4)
    zipdepth_compile_shader(conv2d_spatial4_tiled_oc4 conv2d_spatial4_tiled
        -DNO_BATCH_NORM=1 -DLOCAL_X=16 -DLOCAL_Y=4
        -DOUTPUT_CHANNEL_TILE=4)
    zipdepth_compile_shader(conv2d_spatial4_stride2_tiled conv2d_spatial4_tiled
        -DNO_BATCH_NORM=1 -DLOCAL_X=16 -DLOCAL_Y=4 -DSAMPLE_STRIDE=2)
    zipdepth_compile_shader(conv2d_spatial4_tiled_small conv2d_spatial4_tiled
        -DNO_BATCH_NORM=1 -DLOCAL_X=8 -DLOCAL_Y=4)
    zipdepth_compile_shader(conv2d_spatial4_tiled_relu conv2d_spatial4_tiled
        -DRELU_INPUT=1 -DLOCAL_X=16 -DLOCAL_Y=4)
    zipdepth_compile_shader(precision_pointwise_fp32)
    zipdepth_compile_shader(precision_pointwise_fp16)
    zipdepth_compile_shader(precision_pointwise_fp16_weights)
    zipdepth_compile_shader(precision_pointwise_int8)
    zipdepth_compile_shader(quantize_rows_int8)
    zipdepth_compile_shader(precision_spatial_fp16)
    zipdepth_compile_shader(precision_spatial_int8)
    foreach(SHARED_SHADER reduce_absmax quantize_nchw_int8)
        inferbridge_compile_native_support_shader(
            NAME "${SHARED_SHADER}" PREFIX midas
            GENERATED_DIR "${ZIPDEPTH_GENERATED_DIR}"
            EMBED_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake"
            OUT_VAR ZIPDEPTH_SHADER_HEADERS)
    endforeach()
    inferbridge_compile_native_support_shader(
        NAME conv2d_spatial_int8 SOURCE_NAME conv2d_spatial_int8_tiled
        PREFIX midas GENERATED_DIR "${ZIPDEPTH_GENERATED_DIR}"
        EMBED_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake"
        OUT_VAR ZIPDEPTH_SHADER_HEADERS)
    inferbridge_compile_native_support_shader(
        NAME conv2d_spatial_int8_relu SOURCE_NAME conv2d_spatial_int8_tiled
        PREFIX midas GENERATED_DIR "${ZIPDEPTH_GENERATED_DIR}"
        EMBED_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake"
        OUT_VAR ZIPDEPTH_SHADER_HEADERS EXTRA_ARGUMENTS -DFUSE_RELU=1)
    foreach(SHADER batch_norm_activation activation add bilinear elementwise
            channel_average strip_attention adaptive_average nearest maxpool5
            concat4 context_reduce convex_upsample mobile_upsample
            preprocess_texture resize_depth_image reduce_minmax
            normalize_relative)
        zipdepth_compile_shader(${SHADER})
    endforeach()
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        zipdepth_compile_shader(preprocess_capture)
    endif()
endif()

add_library(zipdepth_native SHARED
    src/zipdepth_native.cpp
    src/inferbridge_harness.cpp
    src/cpu_executor.cpp
    src/model.cpp)
if(ZIPDEPTH_WITH_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "ZIPDEPTH_WITH_METAL requires an Apple platform")
    endif()
    target_sources(zipdepth_native PRIVATE src/metal_executor.mm)
    set_source_files_properties(src/metal_executor.mm PROPERTIES
        COMPILE_OPTIONS "-fobjc-arc")
    target_compile_definitions(zipdepth_native PRIVATE ZIPDEPTH_WITH_METAL=1)
    target_link_libraries(zipdepth_native PRIVATE
        "-framework Foundation" "-framework Metal"
        "-framework MetalPerformanceShadersGraph")
endif()
target_include_directories(zipdepth_native PUBLIC include PRIVATE src)
target_compile_definitions(zipdepth_native PRIVATE
    ZIPDEPTH_BUILD_DLL INFERBRIDGE_HARNESS_BUILDING_LIBRARY)
if(ZIPDEPTH_WITH_VULKAN)
    target_sources(zipdepth_native PRIVATE src/vulkan.cpp src/vulkan_operators.cpp
        src/vulkan_executor.cpp src/zipdepth_ops.cpp src/gpu_io.cpp
        src/external_gpu.cpp
        ${ZIPDEPTH_SHADER_HEADERS})
    target_include_directories(zipdepth_native PRIVATE "${ZIPDEPTH_GENERATED_DIR}")
    target_compile_definitions(zipdepth_native PRIVATE ZIPDEPTH_WITH_VULKAN=1)
    target_link_libraries(zipdepth_native PRIVATE Vulkan::Vulkan)
    if(ANDROID)
        target_link_libraries(zipdepth_native PRIVATE android log)
    endif()
    if(WIN32)
        target_link_libraries(zipdepth_native PRIVATE d3d12 dxgi)
    endif()
endif()
if(MSVC)
    target_compile_options(zipdepth_native PRIVATE /W4 /permissive-)
    set_property(TARGET zipdepth_native PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

if(WIN32 AND ZIPDEPTH_WITH_VULKAN)
    add_executable(zipdepth_d3d12_full_graph tests/d3d12_full_graph.cpp)
    target_include_directories(zipdepth_d3d12_full_graph PRIVATE include src)
    target_link_libraries(zipdepth_d3d12_full_graph PRIVATE
        zipdepth_native d3d12 dxgi)
    if(MSVC)
        target_compile_options(zipdepth_d3d12_full_graph PRIVATE /W4 /permissive-)
        set_property(TARGET zipdepth_d3d12_full_graph PROPERTY
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
endif()

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME zipdepth_android_parity_contract
        COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/android_parity_contract.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/inferbridge_harness.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan_executor.cpp")
endif()
if(BUILD_TESTING AND APPLE AND ZIPDEPTH_WITH_METAL)
    add_executable(zipdepth_metal_full_graph tests/metal_full_graph.cpp)
    target_include_directories(zipdepth_metal_full_graph PRIVATE include)
    target_link_libraries(zipdepth_metal_full_graph PRIVATE zipdepth_native)
    add_test(NAME zipdepth_metal_full_graph COMMAND zipdepth_metal_full_graph)
    set_tests_properties(zipdepth_metal_full_graph PROPERTIES SKIP_RETURN_CODE 77)
endif()
add_executable(zipdepth_model_probe tools/model_probe.cpp src/model.cpp)
target_include_directories(zipdepth_model_probe PRIVATE include src)
if(MSVC)
    target_compile_options(zipdepth_model_probe PRIVATE /W4 /permissive-)
    set_property(TARGET zipdepth_model_probe PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

if(ZIPDEPTH_WITH_VULKAN)
    add_executable(zipdepth_benchmark tools/benchmark.c)
    target_link_libraries(zipdepth_benchmark PRIVATE zipdepth_native)
    target_include_directories(zipdepth_benchmark PRIVATE include)
endif()

if(UNIX)
    add_executable(zipdepth_harness_benchmark tools/harness_benchmark.cpp)
    target_include_directories(zipdepth_harness_benchmark PRIVATE include)
    target_link_libraries(zipdepth_harness_benchmark PRIVATE ${CMAKE_DL_LIBS})
endif()

if(ZIPDEPTH_WITH_VULKAN)
    add_executable(zipdepth_precision_probe
        tools/precision_probe.cpp src/vulkan.cpp ${ZIPDEPTH_SHADER_HEADERS})
    target_include_directories(zipdepth_precision_probe PRIVATE
        include src "${ZIPDEPTH_GENERATED_DIR}")
    target_link_libraries(zipdepth_precision_probe PRIVATE
        Vulkan::Vulkan)
    add_executable(zipdepth_spatial_precision_probe
        tools/spatial_precision_probe.cpp src/vulkan.cpp
        src/vulkan_operators.cpp ${ZIPDEPTH_SHADER_HEADERS})
    target_include_directories(zipdepth_spatial_precision_probe PRIVATE
        include src "${ZIPDEPTH_GENERATED_DIR}")
    target_link_libraries(zipdepth_spatial_precision_probe PRIVATE
        Vulkan::Vulkan)
    if(WIN32)
        target_link_libraries(zipdepth_precision_probe PRIVATE d3d12 dxgi)
        target_link_libraries(zipdepth_spatial_precision_probe PRIVATE
            d3d12 dxgi)
    endif()
endif()
