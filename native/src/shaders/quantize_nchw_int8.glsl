#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) writeonly buffer Packed {
    uint data[];
} packed_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Source {
    float data[];
} source_buffer;
layout(set = 0, binding = 2, std430) readonly buffer Scale {
    float data[];
} scale_buffer;
layout(push_constant) uniform Parameters {
    uint width;
    uint height;
    uint channels;
} parameters;

void main() {
    const uint index = gl_GlobalInvocationID.x;
    const uint packed_channels = parameters.channels / 4u;
    const uint spatial_count = parameters.width * parameters.height;
    if (index >= spatial_count * packed_channels) return;
    const uint spatial = index / packed_channels;
    const uint channel_base = (index % packed_channels) * 4u;
    const float inverse_scale = 1.0 / scale_buffer.data[0];
    uint packed = 0u;
    for (uint lane = 0u; lane < 4u; ++lane) {
        const int value = int(round(clamp(source_buffer.data[
            (channel_base + lane) * spatial_count + spatial] *
            inverse_scale, -127.0, 127.0)));
        packed |= (uint(value) & 0xffu) << (lane * 8u);
    }
    packed_buffer.data[index] = packed;
}
