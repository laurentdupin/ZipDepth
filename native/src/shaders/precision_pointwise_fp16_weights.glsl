#version 450
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input {
    float data[];
} input_buffer;
layout(set = 0, binding = 2, std430) readonly buffer Weight {
    float16_t data[];
} weight_buffer;
layout(set = 0, binding = 3, std430) readonly buffer Bias {
    float data[];
} bias_buffer;
layout(push_constant) uniform Parameters {
    uint spatial;
    uint input_channels;
    uint output_channels;
    uint has_bias;
} parameters;

void main() {
    const uint index = gl_GlobalInvocationID.x;
    const uint count = parameters.spatial * parameters.output_channels;
    if (index >= count) return;
    const uint position = index / parameters.output_channels;
    const uint output_channel = index % parameters.output_channels;
    float sum = 0.0;
    for (uint input_channel = 0;
         input_channel < parameters.input_channels; ++input_channel) {
        sum += input_buffer.data[
            input_channel * parameters.spatial + position] *
            float(weight_buffer.data[
                output_channel * parameters.input_channels + input_channel]);
    }
    output_buffer.data[
        output_channel * parameters.spatial + position] = sum +
        (parameters.has_bias != 0 ? bias_buffer.data[output_channel] : 0.0);
}
