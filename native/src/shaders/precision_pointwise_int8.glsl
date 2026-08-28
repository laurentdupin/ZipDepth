#version 460
#extension GL_EXT_integer_dot_product : require

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
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
layout(push_constant) uniform Parameters {
    uint spatial;
    uint input_channels;
    uint output_channels;
} parameters;

void main() {
    const uint index = gl_GlobalInvocationID.x;
    const uint count = parameters.spatial * parameters.output_channels;
    if (index >= count) return;
    const uint position = index / parameters.output_channels;
    const uint output_channel = index % parameters.output_channels;
    const uint packed_channels = parameters.input_channels / 4;
    int sum = 0;
    for (uint group = 0; group < packed_channels; ++group) {
        sum += dotPacked4x8EXT(
            input_buffer.data[position * packed_channels + group],
            weight_buffer.data[output_channel * packed_channels + group]);
    }
    output_buffer.data[index] = float(sum) *
        input_scale_buffer.data[position] *
        weight_scale_buffer.data[output_channel];
}
