#version 450
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

#define LOCAL_X 16
#define LOCAL_Y 4
#define TILE_WIDTH (LOCAL_X + 2)
#define TILE_HEIGHT (LOCAL_Y + 2)
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
    float16_t data[];
} input_buffer;
layout(set = 0, binding = 2, std430) readonly buffer Weight {
    float16_t data[];
} weight_buffer;
layout(set = 0, binding = 3, std430) readonly buffer Bias {
    float data[];
} bias_buffer;
layout(push_constant) uniform Parameters {
    uint input_width; uint input_height; uint input_channels;
    uint output_width; uint output_height; uint output_channels;
    uint kernel_height; uint kernel_width; uint stride;
    int padding_top; int padding_left; uint dilation; uint groups;
    uint has_bias; uint has_batch_norm; uint activation; float epsilon;
} parameters;

shared float16_t spatial_tile[INPUT_CHANNEL_TILE * TILE_AREA];
shared float16_t kernel_tile[
    INPUT_CHANNEL_TILE * OUTPUT_CHANNEL_TILE * KERNEL_AREA];

void main() {
    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;
    const uint channel_base = gl_GlobalInvocationID.z * OUTPUT_CHANNEL_TILE;
    const bool valid = x < parameters.output_width &&
        y < parameters.output_height &&
        channel_base < parameters.output_channels;
    float sums[OUTPUT_CHANNEL_TILE];
    for (uint offset = 0; offset < OUTPUT_CHANNEL_TILE; ++offset)
        sums[offset] = 0.0;
    const uint lane = gl_LocalInvocationID.y * LOCAL_X +
        gl_LocalInvocationID.x;
    const uint local_count = LOCAL_X * LOCAL_Y;
    const int origin_x = int(gl_WorkGroupID.x * LOCAL_X) - 1;
    const int origin_y = int(gl_WorkGroupID.y * LOCAL_Y) - 1;
    for (uint input_channel_base = 0;
         input_channel_base < parameters.input_channels;
         input_channel_base += INPUT_CHANNEL_TILE) {
        for (uint index = lane; index < INPUT_CHANNEL_TILE * TILE_AREA;
             index += local_count) {
            const uint input_offset = index / TILE_AREA;
            const uint tile_index = index % TILE_AREA;
            const uint input_channel = input_channel_base + input_offset;
            const int input_x = origin_x + int(tile_index % TILE_WIDTH);
            const int input_y = origin_y + int(tile_index / TILE_WIDTH);
            spatial_tile[index] = input_channel < parameters.input_channels &&
                input_x >= 0 && input_x < int(parameters.input_width) &&
                input_y >= 0 && input_y < int(parameters.input_height)
                ? input_buffer.data[
                    (input_channel * parameters.input_height +
                     uint(input_y)) * parameters.input_width + uint(input_x)]
                : float16_t(0.0);
        }
        for (uint index = lane;
             index < INPUT_CHANNEL_TILE * OUTPUT_CHANNEL_TILE * KERNEL_AREA;
             index += local_count) {
            const uint input_offset = index /
                (OUTPUT_CHANNEL_TILE * KERNEL_AREA);
            const uint kernel_index = index %
                (OUTPUT_CHANNEL_TILE * KERNEL_AREA);
            const uint input_channel = input_channel_base + input_offset;
            const uint offset = kernel_index / KERNEL_AREA;
            const uint channel = channel_base + offset;
            kernel_tile[index] = input_channel < parameters.input_channels &&
                channel < parameters.output_channels
                ? weight_buffer.data[
                    (channel * parameters.input_channels + input_channel) *
                    KERNEL_AREA + kernel_index % KERNEL_AREA]
                : float16_t(0.0);
        }
        barrier();
        if (valid) {
            for (uint input_offset = 0;
                 input_offset < INPUT_CHANNEL_TILE &&
                 input_channel_base + input_offset <
                    parameters.input_channels; ++input_offset)
                for (uint kernel_index = 0;
                     kernel_index < KERNEL_AREA; ++kernel_index) {
                    const uint ky = kernel_index / 3;
                    const uint kx = kernel_index % 3;
                    const float16_t value = spatial_tile[
                        input_offset * TILE_AREA +
                        (gl_LocalInvocationID.y + ky) * TILE_WIDTH +
                        gl_LocalInvocationID.x + kx];
                    for (uint offset = 0;
                         offset < OUTPUT_CHANNEL_TILE; ++offset) {
                        const float16_t product = value * kernel_tile[
                            input_offset *
                                (OUTPUT_CHANNEL_TILE * KERNEL_AREA) +
                            offset * KERNEL_AREA + kernel_index];
                        sums[offset] += float(product);
                    }
                }
        }
        barrier();
    }
    if (!valid) return;
    for (uint offset = 0; offset < OUTPUT_CHANNEL_TILE; ++offset) {
        const uint channel = channel_base + offset;
        if (channel < parameters.output_channels)
            output_buffer.data[
                (channel * parameters.output_height + y) *
                parameters.output_width + x] = sums[offset];
    }
}
