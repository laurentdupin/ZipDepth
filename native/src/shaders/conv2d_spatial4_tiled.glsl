#version 450

#ifndef LOCAL_X
#define LOCAL_X 16
#endif
#ifndef LOCAL_Y
#define LOCAL_Y 8
#endif
#ifndef SAMPLE_STRIDE
#define SAMPLE_STRIDE 1
#endif
#define TILE_WIDTH (LOCAL_X * SAMPLE_STRIDE + 2)
#define TILE_HEIGHT (LOCAL_Y * SAMPLE_STRIDE + 2)
#define TILE_AREA (TILE_WIDTH * TILE_HEIGHT)
#define INPUT_CHANNEL_TILE 8
#define OUTPUT_CHANNEL_TILE 8
#define KERNEL_AREA 9

layout(local_size_x = LOCAL_X, local_size_y = LOCAL_Y,
       local_size_z = 1) in;
layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input {
    float data[];
} input_buffer;
layout(set = 0, binding = 2, std430) readonly buffer Weight {
    float data[];
} weight_buffer;
layout(set = 0, binding = 3, std430) readonly buffer Bias {
    float data[];
} bias_buffer;
#if !defined(NO_BATCH_NORM) && !defined(RELU_INPUT)
layout(set = 0, binding = 4, std430) readonly buffer Gamma { float data[]; } gamma_buffer;
layout(set = 0, binding = 5, std430) readonly buffer Beta { float data[]; } beta_buffer;
layout(set = 0, binding = 6, std430) readonly buffer Mean { float data[]; } mean_buffer;
layout(set = 0, binding = 7, std430) readonly buffer Variance { float data[]; } variance_buffer;
#endif
layout(push_constant) uniform Parameters {
    uint input_width; uint input_height; uint input_channels;
    uint output_width; uint output_height; uint output_channels;
    uint kernel_height; uint kernel_width; uint stride;
    int padding_top; int padding_left; uint dilation; uint groups;
    uint has_bias;
    uint has_batch_norm; uint activation; float epsilon;
} parameters;
shared float spatial_tile[INPUT_CHANNEL_TILE * TILE_AREA];
shared float kernel_tile[
    INPUT_CHANNEL_TILE * OUTPUT_CHANNEL_TILE * KERNEL_AREA];

float transform_input(float value) {
#if defined(RELU_INPUT)
    return max(value, 0.0);
#else
    return value;
#endif
}

void main() {
    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;
    const uint channel_base = gl_GlobalInvocationID.z * OUTPUT_CHANNEL_TILE;
    const bool valid =
        x < parameters.output_width &&
        y < parameters.output_height &&
        channel_base < parameters.output_channels;
    float sums[OUTPUT_CHANNEL_TILE];
    for (uint offset = 0; offset < OUTPUT_CHANNEL_TILE; ++offset)
        sums[offset] = 0.0;
    const uint lane = gl_LocalInvocationID.y * LOCAL_X +
        gl_LocalInvocationID.x;
    const uint local_count = LOCAL_X * LOCAL_Y;
    const int origin_x =
        int(gl_WorkGroupID.x * LOCAL_X * SAMPLE_STRIDE) - 1;
    const int origin_y =
        int(gl_WorkGroupID.y * LOCAL_Y * SAMPLE_STRIDE) - 1;
    for (uint input_channel_base = 0;
         input_channel_base < parameters.input_channels;
         input_channel_base += INPUT_CHANNEL_TILE) {
        for (uint index = lane;
             index < INPUT_CHANNEL_TILE * TILE_AREA;
             index += local_count) {
            const uint input_offset = index / TILE_AREA;
            const uint tile_index = index % TILE_AREA;
            const uint input_channel =
                input_channel_base + input_offset;
            const int input_x = origin_x + int(tile_index % TILE_WIDTH);
            const int input_y = origin_y + int(tile_index / TILE_WIDTH);
            spatial_tile[index] =
                input_channel < parameters.input_channels &&
                input_x >= 0 && input_x < int(parameters.input_width) &&
                input_y >= 0 && input_y < int(parameters.input_height)
                ? transform_input(input_buffer.data[
                      (input_channel * parameters.input_height +
                       uint(input_y)) * parameters.input_width +
                      uint(input_x)])
                : 0.0;
        }
        for (uint index = lane;
             index < INPUT_CHANNEL_TILE * OUTPUT_CHANNEL_TILE * KERNEL_AREA;
             index += local_count) {
            const uint input_offset = index /
                (OUTPUT_CHANNEL_TILE * KERNEL_AREA);
            const uint kernel_index = index %
                (OUTPUT_CHANNEL_TILE * KERNEL_AREA);
            const uint input_channel =
                input_channel_base + input_offset;
            const uint offset = kernel_index / 9;
            const uint channel = channel_base + offset;
            kernel_tile[index] =
                input_channel < parameters.input_channels &&
                channel < parameters.output_channels
                ? weight_buffer.data[
                      (channel * parameters.input_channels + input_channel) *
                      KERNEL_AREA + kernel_index % KERNEL_AREA]
                : 0.0;
        }
        barrier();
        if (valid) {
            for (uint input_offset = 0;
                 input_offset < INPUT_CHANNEL_TILE &&
                 input_channel_base + input_offset <
                     parameters.input_channels;
                 ++input_offset)
                for (uint ky = 0; ky < 3; ++ky)
                    for (uint kx = 0; kx < 3; ++kx) {
                        const uint kernel_index = ky * 3 + kx;
                        const float value = spatial_tile[
                            input_offset * TILE_AREA +
                            (gl_LocalInvocationID.y * SAMPLE_STRIDE + ky) *
                                TILE_WIDTH +
                            gl_LocalInvocationID.x * SAMPLE_STRIDE + kx];
                        for (uint offset = 0;
                             offset < OUTPUT_CHANNEL_TILE; ++offset)
                            sums[offset] += value * kernel_tile[
                                input_offset *
                                    (OUTPUT_CHANNEL_TILE * KERNEL_AREA) +
                                offset * KERNEL_AREA +
                                kernel_index];
                    }
        }
        barrier();
    }
    if (!valid) return;
    for (uint offset = 0; offset < OUTPUT_CHANNEL_TILE; ++offset) {
        const uint channel = channel_base + offset;
        if (channel < parameters.output_channels) {
            float value = sums[offset] + (parameters.has_bias != 0
                ? bias_buffer.data[channel] : 0.0);
#if !defined(NO_BATCH_NORM) && !defined(RELU_INPUT)
            if (parameters.has_batch_norm != 0) {
                value = value * gamma_buffer.data[channel] +
                    beta_buffer.data[channel];
                if (parameters.activation == 1) value = max(value, 0.0);
                else if (parameters.activation == 2)
                    value = clamp(value, 0.0, 6.0);
            } else
#endif
            if (parameters.activation == 1) {
                value = max(value, 0.0);
            } else if (parameters.activation == 2) {
                value = clamp(value, 0.0, 6.0);
            }
            output_buffer.data[
                (channel * parameters.output_height + y) *
                parameters.output_width + x] =
                value;
        }
    }
}
