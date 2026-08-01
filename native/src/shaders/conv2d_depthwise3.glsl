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
#if !defined(NO_BATCH_NORM)
layout(set = 0, binding = 4, std430) readonly buffer Gamma {
    float data[];
} gamma_buffer;
layout(set = 0, binding = 5, std430) readonly buffer Beta {
    float data[];
} beta_buffer;
layout(set = 0, binding = 6, std430) readonly buffer Mean {
    float data[];
} mean_buffer;
layout(set = 0, binding = 7, std430) readonly buffer Variance {
    float data[];
} variance_buffer;
#endif
layout(push_constant) uniform Parameters {
    uint input_width; uint input_height; uint input_channels;
    uint output_width; uint output_height; uint output_channels;
    uint kernel_height; uint kernel_width; uint stride;
    int padding_top; int padding_left; uint groups; uint has_bias;
    uint has_batch_norm; uint activation; float epsilon;
} parameters;
shared float tile[18 * 18];

void main() {
    const uint channel = gl_GlobalInvocationID.z;
    const uint lane = gl_LocalInvocationID.y * 8 +
        gl_LocalInvocationID.x;
    const uint tile_width = 8 * parameters.stride + 2;
    const uint tile_height = 8 * parameters.stride + 2;
    const int origin_x =
        int(gl_WorkGroupID.x * 8 * parameters.stride) -
        parameters.padding_left;
    const int origin_y =
        int(gl_WorkGroupID.y * 8 * parameters.stride) -
        parameters.padding_top;
    for (uint index = lane;
         index < tile_width * tile_height;
         index += 64) {
        const int x = origin_x + int(index % tile_width);
        const int y = origin_y + int(index / tile_width);
        tile[index] =
            channel < parameters.input_channels &&
            x >= 0 && x < int(parameters.input_width) &&
            y >= 0 && y < int(parameters.input_height)
            ? input_buffer.data[
                  (channel * parameters.input_height + uint(y)) *
                  parameters.input_width + uint(x)]
            : 0.0;
    }
    barrier();
    const uint output_x = gl_GlobalInvocationID.x;
    const uint output_y = gl_GlobalInvocationID.y;
    if (channel >= parameters.output_channels ||
        output_x >= parameters.output_width ||
        output_y >= parameters.output_height) return;
    const uint local_x =
        gl_LocalInvocationID.x * parameters.stride;
    const uint local_y =
        gl_LocalInvocationID.y * parameters.stride;
    float sum = parameters.has_bias != 0
        ? bias_buffer.data[channel] : 0.0;
    for (uint ky = 0; ky < 3; ++ky)
        for (uint kx = 0; kx < 3; ++kx)
            sum += tile[(local_y + ky) * tile_width + local_x + kx] *
                weight_buffer.data[channel * 9 + ky * 3 + kx];
#if !defined(NO_BATCH_NORM)
    if (parameters.has_batch_norm != 0) {
        sum = sum * gamma_buffer.data[channel] +
            beta_buffer.data[channel];
        if (parameters.activation == 1)
            sum = max(sum, 0.0);
        else if (parameters.activation == 2)
            sum = clamp(sum, 0.0, 6.0);
    } else
#endif
    if (parameters.activation == 1) {
        sum = max(sum, 0.0);
    } else if (parameters.activation == 2) {
        sum = clamp(sum, 0.0, 6.0);
    }
    output_buffer.data[
        (channel * parameters.output_height + output_y) *
        parameters.output_width + output_x] = sum;
}
