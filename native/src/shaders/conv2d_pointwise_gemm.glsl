#version 450

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
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
#if !defined(NO_BATCH_NORM) && !defined(ADD_RESIDUAL)
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
#if defined(ADD_RESIDUAL)
layout(set = 0, binding = 4, std430) readonly buffer Residual {
    float data[];
} residual_buffer;
#endif
layout(push_constant) uniform Parameters {
    uint input_width; uint input_height; uint input_channels;
    uint output_width; uint output_height; uint output_channels;
    uint kernel_height; uint kernel_width; uint stride;
    int padding_top; int padding_left; uint groups; uint has_bias;
    uint has_batch_norm; uint activation; float epsilon;
} parameters;

#define K_TILE 32
#define K_STRIDE (K_TILE + 1)
shared float input_tile[64 * K_STRIDE];
shared float weight_tile[64 * K_STRIDE];

void main() {
    const uint spatial_count =
        parameters.input_width * parameters.input_height;
    const uint spatial_base =
        gl_WorkGroupID.x * 64 + gl_LocalInvocationID.x * 4;
    const uint channel_base =
        gl_WorkGroupID.y * 64 + gl_LocalInvocationID.y * 4;
    const uint lane =
        gl_LocalInvocationID.y * gl_WorkGroupSize.x +
        gl_LocalInvocationID.x;
    float sums[4][4];
    for (uint channel = 0; channel < 4; ++channel)
        for (uint spatial = 0; spatial < 4; ++spatial)
            sums[channel][spatial] = 0.0;

    for (uint input_base = 0;
         input_base < parameters.input_channels;
         input_base += K_TILE) {
        for (uint index = lane; index < 64 * K_TILE; index += 256) {
            const uint spatial_offset = index / K_TILE;
            const uint input_channel = input_base + index % K_TILE;
            const uint spatial = gl_WorkGroupID.x * 64 + spatial_offset;
            input_tile[spatial_offset * K_STRIDE +
                       index % K_TILE] =
                spatial < spatial_count &&
                input_channel < parameters.input_channels
                ? input_buffer.data[input_channel * spatial_count + spatial]
                : 0.0;
        }
        for (uint index = lane; index < 64 * K_TILE; index += 256) {
            const uint channel_offset = index / K_TILE;
            const uint input_channel = input_base + index % K_TILE;
            const uint channel =
                gl_WorkGroupID.y * 64 + channel_offset;
            weight_tile[channel_offset * K_STRIDE +
                        index % K_TILE] =
                channel < parameters.output_channels &&
                input_channel < parameters.input_channels
                ? weight_buffer.data[
                      channel * parameters.input_channels + input_channel]
                : 0.0;
        }
        barrier();
        const uint count =
            min(K_TILE, parameters.input_channels - input_base);
        for (uint inner = 0; inner < count; ++inner) {
            float inputs[4];
            float weights[4];
            for (uint spatial = 0; spatial < 4; ++spatial)
                inputs[spatial] = input_tile[
                    (gl_LocalInvocationID.x * 4 + spatial) * K_STRIDE +
                    inner];
            for (uint channel = 0; channel < 4; ++channel)
                weights[channel] = weight_tile[
                    (gl_LocalInvocationID.y * 4 + channel) * K_STRIDE +
                    inner];
            for (uint channel = 0; channel < 4; ++channel)
                for (uint spatial = 0; spatial < 4; ++spatial)
                    sums[channel][spatial] +=
                        weights[channel] * inputs[spatial];
        }
        barrier();
    }

    for (uint channel_offset = 0; channel_offset < 4; ++channel_offset) {
        const uint channel = channel_base + channel_offset;
        if (channel >= parameters.output_channels) continue;
        const float bias = parameters.has_bias != 0
            ? bias_buffer.data[channel] : 0.0;
        for (uint spatial_offset = 0; spatial_offset < 4; ++spatial_offset) {
            const uint spatial = spatial_base + spatial_offset;
            if (spatial < spatial_count) {
                float value = sums[channel_offset][spatial_offset] + bias;
#if !defined(NO_BATCH_NORM) && !defined(ADD_RESIDUAL)
                if (parameters.has_batch_norm != 0) {
                    value = value * gamma_buffer.data[channel] +
                        beta_buffer.data[channel];
                    if (parameters.activation == 1)
                        value = max(value, 0.0);
                    else if (parameters.activation == 2)
                        value = clamp(value, 0.0, 6.0);
                } else
#endif
                if (parameters.activation == 1) {
                    value = max(value, 0.0);
                } else if (parameters.activation == 2) {
                    value = clamp(value, 0.0, 6.0);
                }
#if defined(ADD_RESIDUAL)
                value += residual_buffer.data[
                    channel * spatial_count + spatial];
#endif
                output_buffer.data[channel * spatial_count + spatial] =
                    value;
            }
        }
    }
}
