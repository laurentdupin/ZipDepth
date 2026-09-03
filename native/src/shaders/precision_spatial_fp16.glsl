#version 450
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = 16, local_size_y = 4, local_size_z = 1) in;
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
    uint has_bias;
    uint has_batch_norm; uint activation; float epsilon;
} parameters;

#define TILE_WIDTH 18
#define TILE_HEIGHT 6
#define TILE_AREA (TILE_WIDTH * TILE_HEIGHT)
#define INPUT_CHANNEL_TILE 8
#define OUTPUT_CHANNEL_TILE 8
#define KERNEL_AREA 9

shared float spatial_tile[INPUT_CHANNEL_TILE * TILE_AREA];
shared float kernel_tile[
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

    const uint lane = gl_LocalInvocationID.y * 16 +
        gl_LocalInvocationID.x;
    const int origin_x = int(gl_WorkGroupID.x * 16) - 1;
    const int origin_y = int(gl_WorkGroupID.y * 4) - 1;
    for (uint input_base = 0;
         input_base < parameters.input_channels;
         input_base += INPUT_CHANNEL_TILE) {
        for (uint index = lane;
             index < INPUT_CHANNEL_TILE * TILE_AREA;
             index += 64) {
            const uint input_offset = index / TILE_AREA;
            const uint tile_index = index % TILE_AREA;
            const uint input_channel = input_base + input_offset;
            const int input_x = origin_x + int(tile_index % TILE_WIDTH);
            const int input_y = origin_y + int(tile_index / TILE_WIDTH);
            spatial_tile[index] = input_channel < parameters.input_channels &&
                input_x >= 0 && input_x < int(parameters.input_width) &&
                input_y >= 0 && input_y < int(parameters.input_height)
                ? float(input_buffer.data[
                    (input_channel * parameters.input_height +
                     uint(input_y)) * parameters.input_width + uint(input_x)])
                : 0.0;
        }
        for (uint index = lane;
             index < INPUT_CHANNEL_TILE * OUTPUT_CHANNEL_TILE * KERNEL_AREA;
             index += 64) {
            const uint input_offset = index /
                (OUTPUT_CHANNEL_TILE * KERNEL_AREA);
            const uint kernel_index = index %
                (OUTPUT_CHANNEL_TILE * KERNEL_AREA);
            const uint input_channel = input_base + input_offset;
            const uint channel = channel_base + kernel_index / KERNEL_AREA;
            kernel_tile[index] = input_channel < parameters.input_channels &&
                channel < parameters.output_channels
                ? float(weight_buffer.data[
                    (channel * parameters.input_channels + input_channel) *
                    KERNEL_AREA + kernel_index % KERNEL_AREA])
                : 0.0;
        }
        barrier();
        if (valid) {
            const uint count = min(
                INPUT_CHANNEL_TILE, parameters.input_channels - input_base);
            for (uint input_offset = 0; input_offset < count; ++input_offset)
                for (uint ky = 0; ky < 3; ++ky)
                    for (uint kx = 0; kx < 3; ++kx) {
                        const uint kernel = ky * 3 + kx;
                        const float value = spatial_tile[
                            input_offset * TILE_AREA +
                            (gl_LocalInvocationID.y + ky) * TILE_WIDTH +
                            gl_LocalInvocationID.x + kx];
                        for (uint offset = 0;
                             offset < OUTPUT_CHANNEL_TILE; ++offset)
                            sums[offset] += value * kernel_tile[
                                input_offset *
                                    (OUTPUT_CHANNEL_TILE * KERNEL_AREA) +
                                offset * KERNEL_AREA + kernel];
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
                parameters.output_width + x] = sums[offset] +
                (parameters.has_bias != 0 ? bias_buffer.data[channel] : 0.0);
    }
}
