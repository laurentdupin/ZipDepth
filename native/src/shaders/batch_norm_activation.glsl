#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input {
    float data[];
} input_buffer;
layout(set = 0, binding = 2, std430) readonly buffer Gamma {
    float data[];
} gamma_buffer;
layout(set = 0, binding = 3, std430) readonly buffer Beta {
    float data[];
} beta_buffer;
layout(set = 0, binding = 4, std430) readonly buffer Mean {
    float data[];
} mean_buffer;
layout(set = 0, binding = 5, std430) readonly buffer Variance {
    float data[];
} variance_buffer;

layout(push_constant) uniform Parameters {
    uint count;
    uint plane;
    uint activation;
    float epsilon;
} parameters;

void main() {
    const uint index = gl_GlobalInvocationID.x;
    if (index >= parameters.count) return;
    const uint channel = index / parameters.plane;
    const float scale = gamma_buffer.data[channel] *
        inversesqrt(variance_buffer.data[channel] + parameters.epsilon);
    float value = (input_buffer.data[index] -
        mean_buffer.data[channel]) * scale + beta_buffer.data[channel];
    if (parameters.activation == 1) {
        value = max(value, 0.0);
    } else if (parameters.activation == 2) {
        value = clamp(value, 0.0, 6.0);
    }
    output_buffer.data[index] = value;
}
