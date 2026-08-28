#version 450
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input {
    vec4 data[];
} input_buffer;
layout(set = 0, binding = 2, std430) readonly buffer Weight {
    f16vec4 data[];
} weight_buffer;
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
    float sum = 0.0;
    const uint packed_channels = parameters.input_channels / 4;
    for (uint group = 0; group < packed_channels; ++group) {
        sum += dot(input_buffer.data[
            position * packed_channels + group], vec4(weight_buffer.data[
                output_channel * packed_channels + group]));
    }
    output_buffer.data[index] = sum;
}
