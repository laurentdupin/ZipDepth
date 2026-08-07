#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
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
layout(push_constant) uniform Parameters {
    uint input_width; uint input_height; uint input_channels;
    uint output_width; uint output_height; uint output_channels;
    uint kernel_height; uint kernel_width; uint stride;
    int padding_top; int padding_left; uint dilation; uint groups;
    uint has_bias;
    uint has_batch_norm; uint activation; float epsilon;
} parameters;

#define INPUT_TILE 64
#define OUTPUT_TILE 4
shared float weight_tile[OUTPUT_TILE * INPUT_TILE];

void main() {
    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;
    const uint channel_base = gl_GlobalInvocationID.z * OUTPUT_TILE;
    const bool valid_pixel =
        x < parameters.output_width && y < parameters.output_height &&
        channel_base < parameters.output_channels;
    float sums[OUTPUT_TILE];
    for (uint offset = 0; offset < OUTPUT_TILE; ++offset)
        sums[offset] = 0.0;
    const uint input_plane = parameters.input_width *
        parameters.input_height;
    const uint output_plane = parameters.output_width *
        parameters.output_height;
    const uint spatial = y * parameters.input_width + x;
    const uint lane = gl_LocalInvocationID.y * 8 +
        gl_LocalInvocationID.x;
    for (uint input_base = 0; input_base < parameters.input_channels;
         input_base += INPUT_TILE) {
        for (uint index = lane;
             index < OUTPUT_TILE * INPUT_TILE; index += 64) {
            const uint offset = index / INPUT_TILE;
            const uint input_channel = input_base + index % INPUT_TILE;
            const uint channel = channel_base + offset;
            weight_tile[index] =
                channel < parameters.output_channels &&
                input_channel < parameters.input_channels
                ? weight_buffer.data[
                    channel * parameters.input_channels + input_channel]
                : 0.0;
        }
        barrier();
        if (valid_pixel) {
            const uint count = min(
                INPUT_TILE, parameters.input_channels - input_base);
            for (uint inner = 0; inner < count; ++inner) {
                const float value = input_buffer.data[
                    (input_base + inner) * input_plane + spatial];
                for (uint offset = 0; offset < OUTPUT_TILE; ++offset)
                    sums[offset] +=
                        value * weight_tile[offset * INPUT_TILE + inner];
            }
        }
        barrier();
    }
    if (!valid_pixel) return;
    for (uint offset = 0; offset < OUTPUT_TILE; ++offset) {
        const uint channel = channel_base + offset;
        if (channel < parameters.output_channels)
            output_buffer.data[channel * output_plane +
                y * parameters.output_width + x] =
                sums[offset] + (parameters.has_bias != 0
                    ? bias_buffer.data[channel] : 0.0);
    }
}
