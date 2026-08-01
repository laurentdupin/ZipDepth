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
    int padding_top; int padding_left; uint groups; uint has_bias;
    uint has_batch_norm; uint activation; float epsilon;
} parameters;

void main() {
    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;
    const uint channel_base = gl_GlobalInvocationID.z * 4;
    if (x >= parameters.output_width ||
        y >= parameters.output_height ||
        channel_base >= parameters.output_channels) return;
    float sums[4] = float[4](0.0, 0.0, 0.0, 0.0);
    const uint input_plane = parameters.input_width *
        parameters.input_height;
    const uint output_plane = parameters.output_width *
        parameters.output_height;
    const uint spatial = y * parameters.input_width + x;
    for (uint input_channel = 0;
         input_channel < parameters.input_channels;
         ++input_channel) {
        const float value = input_buffer.data[
            input_channel * input_plane + spatial];
        for (uint offset = 0; offset < 4; ++offset) {
            const uint channel = channel_base + offset;
            if (channel < parameters.output_channels)
                sums[offset] += value * weight_buffer.data[
                    channel * parameters.input_channels + input_channel];
        }
    }
    for (uint offset = 0; offset < 4; ++offset) {
        const uint channel = channel_base + offset;
        if (channel < parameters.output_channels)
            output_buffer.data[channel * output_plane +
                y * parameters.output_width + x] =
                sums[offset] + (parameters.has_bias != 0
                    ? bias_buffer.data[channel] : 0.0);
    }
}
