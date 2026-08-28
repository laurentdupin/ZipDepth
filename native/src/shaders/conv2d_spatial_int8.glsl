#version 460
#extension GL_EXT_integer_dot_product : require

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input {
    int data[];
} input_buffer;
layout(set = 0, binding = 2, std430) readonly buffer Weight {
    int data[];
} weight_buffer;
layout(set = 0, binding = 3, std430) readonly buffer InputScale {
    float data[];
} input_scale_buffer;
layout(set = 0, binding = 4, std430) readonly buffer WeightScale {
    float data[];
} weight_scale_buffer;
layout(set = 0, binding = 5, std430) readonly buffer Bias {
    float data[];
} bias_buffer;
layout(push_constant) uniform Parameters {
    uint width;
    uint height;
    uint input_channels;
    uint output_channels;
    uint has_bias;
} parameters;

void main() {
    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;
    const uint output_channel = gl_GlobalInvocationID.z;
    if (x >= parameters.width || y >= parameters.height ||
        output_channel >= parameters.output_channels) return;
    const uint packed_channels = parameters.input_channels / 4u;
    int sum = 0;
    for (uint ky = 0; ky < 3; ++ky) {
        const int input_y = int(y) + int(ky) - 1;
        if (input_y < 0 || input_y >= int(parameters.height)) continue;
        for (uint kx = 0; kx < 3; ++kx) {
            const int input_x = int(x) + int(kx) - 1;
            if (input_x < 0 || input_x >= int(parameters.width)) continue;
            for (uint group = 0; group < packed_channels; ++group)
                sum += dotPacked4x8EXT(input_buffer.data[
                    (uint(input_y) * parameters.width + uint(input_x)) *
                    packed_channels + group], weight_buffer.data[
                    ((output_channel * 3u + ky) * 3u + kx) *
                    packed_channels + group]);
        }
    }
    float value = float(sum) * input_scale_buffer.data[0] *
        weight_scale_buffer.data[output_channel];
    if (parameters.has_bias != 0u)
        value += bias_buffer.data[output_channel];
    output_buffer.data[
        (output_channel * parameters.height + y) * parameters.width + x] =
        value;
}
