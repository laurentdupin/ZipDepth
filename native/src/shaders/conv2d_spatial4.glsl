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
    uint input_width; uint input_height; uint input_channels;
    uint output_width; uint output_height; uint output_channels;
    uint kernel_height; uint kernel_width; uint stride;
    int padding_top; int padding_left; uint dilation; uint groups;
    uint has_bias;
    uint has_batch_norm; uint activation; float epsilon;
} parameters;

#if defined(STRIDE2_DIRECT)
void accumulate_sample(inout vec4 sums, float value, uint wi, uint channel_stride) {
    sums += value * vec4(weight_buffer.data[wi],
        weight_buffer.data[wi + channel_stride],
        weight_buffer.data[wi + 2 * channel_stride],
        weight_buffer.data[wi + 3 * channel_stride]);
}
#endif

void main() {
    const uint output_x = gl_GlobalInvocationID.x;
    const uint output_y = gl_GlobalInvocationID.y;
    const uint channel_base = gl_GlobalInvocationID.z * 4;
    if (output_x >= parameters.output_width ||
        output_y >= parameters.output_height ||
        channel_base >= parameters.output_channels) return;
    float sums[4] = float[4](0.0, 0.0, 0.0, 0.0);
#if defined(STRIDE2_DIRECT)
    // This pipeline is selected only for 3x3, stride 2, pad 1, four complete
    // output channels. Border pixels retain the generic path below.
    const uint ix = output_x * 2;
    const uint iy = output_y * 2;
    if (ix > 0 && iy > 0 && ix + 1 < parameters.input_width && iy + 1 < parameters.input_height) {
        vec4 direct_sums = vec4(0.0);
        const uint plane = parameters.input_width * parameters.input_height;
        const uint channel_stride = parameters.input_channels * 9;
        uint input_index = (iy - 1) * parameters.input_width + ix - 1;
        uint wi = channel_base * channel_stride;
        for (uint ic = 0; ic < parameters.input_channels; ++ic) {
            accumulate_sample(direct_sums, input_buffer.data[input_index], wi, channel_stride);
            accumulate_sample(direct_sums, input_buffer.data[input_index + 1], wi + 1, channel_stride);
            accumulate_sample(direct_sums, input_buffer.data[input_index + 2], wi + 2, channel_stride);
            accumulate_sample(direct_sums, input_buffer.data[input_index + parameters.input_width], wi + 3, channel_stride);
            accumulate_sample(direct_sums, input_buffer.data[input_index + parameters.input_width + 1], wi + 4, channel_stride);
            accumulate_sample(direct_sums, input_buffer.data[input_index + parameters.input_width + 2], wi + 5, channel_stride);
            accumulate_sample(direct_sums, input_buffer.data[input_index + 2 * parameters.input_width], wi + 6, channel_stride);
            accumulate_sample(direct_sums, input_buffer.data[input_index + 2 * parameters.input_width + 1], wi + 7, channel_stride);
            accumulate_sample(direct_sums, input_buffer.data[input_index + 2 * parameters.input_width + 2], wi + 8, channel_stride);
            input_index += plane;
            wi += 9;
        }
        for (uint i = 0; i < 4; ++i) sums[i] = direct_sums[i];
    } else
#endif
    {
    for (uint input_channel = 0;
         input_channel < parameters.input_channels;
         ++input_channel) {
        for (uint ky = 0; ky < parameters.kernel_height; ++ky) {
            const int input_y =
                int(output_y * parameters.stride + ky) -
                parameters.padding_top;
            if (input_y < 0 ||
                input_y >= int(parameters.input_height)) continue;
            for (uint kx = 0; kx < parameters.kernel_width; ++kx) {
                const int input_x =
                    int(output_x * parameters.stride + kx) -
                    parameters.padding_left;
                if (input_x < 0 ||
                    input_x >= int(parameters.input_width)) continue;
                const float value = input_buffer.data[
                    (input_channel * parameters.input_height +
                     uint(input_y)) * parameters.input_width +
                    uint(input_x)];
                for (uint offset = 0; offset < 4; ++offset) {
                    const uint channel = channel_base + offset;
                    if (channel < parameters.output_channels) {
                        const uint weight_index =
                            ((channel * parameters.input_channels +
                              input_channel) * parameters.kernel_height + ky) *
                            parameters.kernel_width + kx;
                        sums[offset] += value *
                            weight_buffer.data[weight_index];
                    }
                }
            }
        }
    }
    }
    for (uint offset = 0; offset < 4; ++offset) {
        const uint channel = channel_base + offset;
        if (channel < parameters.output_channels) {
            float value = sums[offset] + (parameters.has_bias != 0
                ? bias_buffer.data[channel] : 0.0);
#if !defined(NO_BATCH_NORM)
            if (parameters.has_batch_norm != 0) {
                value = value * gamma_buffer.data[channel] +
                    beta_buffer.data[channel];
                if (parameters.activation == 1) value = max(value, 0.0);
                else if (parameters.activation == 2)
                    value = clamp(value, 0.0, 6.0);
            } else
#endif
            if (parameters.activation == 1) {
                value = max(value, 0.0);
            } else if (parameters.activation == 2) {
                value = clamp(value, 0.0, 6.0);
            }
            output_buffer.data[
                (channel * parameters.output_height + output_y) *
                parameters.output_width + output_x] =
                value;
        }
    }
}
