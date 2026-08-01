#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input {
    float data[];
} input_buffer;
layout(push_constant) uniform Parameters {
    uint input_width;
    uint input_height;
    uint output_width;
    uint output_height;
    uint channels;
    uint align_corners;
} parameters;

void main() {
    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;
    const uint channel = gl_GlobalInvocationID.z;
    if (x >= parameters.output_width ||
        y >= parameters.output_height ||
        channel >= parameters.channels) return;
    float source_x;
    float source_y;
    if (parameters.align_corners != 0) {
        source_x = parameters.output_width > 1
            ? float(x) * float(parameters.input_width - 1) /
                float(parameters.output_width - 1) : 0.0;
        source_y = parameters.output_height > 1
            ? float(y) * float(parameters.input_height - 1) /
                float(parameters.output_height - 1) : 0.0;
    } else {
        source_x = (float(x) + 0.5) *
            float(parameters.input_width) / float(parameters.output_width) -
            0.5;
        source_y = (float(y) + 0.5) *
            float(parameters.input_height) / float(parameters.output_height) -
            0.5;
    }
    source_x = clamp(source_x, 0.0, float(parameters.input_width - 1));
    source_y = clamp(source_y, 0.0, float(parameters.input_height - 1));
    const uint x0 = uint(floor(source_x));
    const uint y0 = uint(floor(source_y));
    const uint x1 = min(x0 + 1, parameters.input_width - 1);
    const uint y1 = min(y0 + 1, parameters.input_height - 1);
    const float fx = source_x - float(x0);
    const float fy = source_y - float(y0);
    const uint plane = parameters.input_width * parameters.input_height;
    const uint base = channel * plane;
    const float top = mix(
        input_buffer.data[base + y0 * parameters.input_width + x0],
        input_buffer.data[base + y0 * parameters.input_width + x1], fx);
    const float bottom = mix(
        input_buffer.data[base + y1 * parameters.input_width + x0],
        input_buffer.data[base + y1 * parameters.input_width + x1], fx);
    output_buffer.data[
        (channel * parameters.output_height + y) *
            parameters.output_width + x] = mix(top, bottom, fy);
}
