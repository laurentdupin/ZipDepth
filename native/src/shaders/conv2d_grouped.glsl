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
layout(set = 0, binding = 4, std430) readonly buffer Gamma { float data[]; } gamma_buffer;
layout(set = 0, binding = 5, std430) readonly buffer Beta { float data[]; } beta_buffer;
layout(set = 0, binding = 6, std430) readonly buffer Mean { float data[]; } mean_buffer;
layout(set = 0, binding = 7, std430) readonly buffer Variance { float data[]; } variance_buffer;
#endif

layout(push_constant) uniform Parameters {
    uint input_width;
    uint input_height;
    uint input_channels;
    uint output_width;
    uint output_height;
    uint output_channels;
    uint kernel_height;
    uint kernel_width;
    uint stride;
    int padding_top;
    int padding_left;
    uint dilation;
    uint groups;
    uint has_bias;
    uint has_batch_norm;
    uint activation;
    float epsilon;
} parameters;

void main() {
    const uint ox = gl_GlobalInvocationID.x;
    const uint oy = gl_GlobalInvocationID.y;
    const uint oc = gl_GlobalInvocationID.z;
    if (ox >= parameters.output_width ||
        oy >= parameters.output_height ||
        oc >= parameters.output_channels) {
        return;
    }
    const uint input_per_group =
        parameters.input_channels / parameters.groups;
    const uint output_per_group =
        parameters.output_channels / parameters.groups;
    const uint input_begin = (oc / output_per_group) * input_per_group;
    float sum = parameters.has_bias != 0
        ? bias_buffer.data[oc] : 0.0;
    for (uint icg = 0; icg < input_per_group; ++icg) {
        const uint ic = input_begin + icg;
        for (uint ky = 0; ky < parameters.kernel_height; ++ky) {
            const int iy = int(oy * parameters.stride + ky * parameters.dilation) -
                parameters.padding_top;
            if (iy < 0 || iy >= int(parameters.input_height)) continue;
            for (uint kx = 0; kx < parameters.kernel_width; ++kx) {
                const int ix = int(ox * parameters.stride + kx * parameters.dilation) -
                    parameters.padding_left;
                if (ix < 0 || ix >= int(parameters.input_width)) continue;
                const uint input_index =
                    (ic * parameters.input_height + uint(iy)) *
                        parameters.input_width + uint(ix);
                const uint weight_index =
                    ((oc * input_per_group + icg) *
                        parameters.kernel_height + ky) *
                        parameters.kernel_width + kx;
                sum += input_buffer.data[input_index] *
                    weight_buffer.data[weight_index];
            }
        }
    }
#if !defined(NO_BATCH_NORM)
    if (parameters.has_batch_norm != 0) {
        sum = sum * gamma_buffer.data[oc] + beta_buffer.data[oc];
        if (parameters.activation == 1) sum = max(sum, 0.0);
        else if (parameters.activation == 2) sum = clamp(sum, 0.0, 6.0);
    } else
#endif
    if (parameters.activation == 1) {
        sum = max(sum, 0.0);
    } else if (parameters.activation == 2) {
        sum = clamp(sum, 0.0, 6.0);
    }
    output_buffer.data[
        (oc * parameters.output_height + oy) *
            parameters.output_width + ox] = sum;
}
