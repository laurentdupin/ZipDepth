#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) readonly buffer Source {
    float data[];
} source_buffer;
layout(set = 0, binding = 1, std430) writeonly buffer Scale {
    float data[];
} scale_buffer;
layout(push_constant) uniform Parameters {
    uint count;
    float divisor;
} parameters;
shared float maxima[256];

void main() {
    const uint lane = gl_LocalInvocationID.x;
    float maximum = 0.0;
    for (uint index = gl_WorkGroupID.x * 256u + lane;
         index < parameters.count;
         index += gl_NumWorkGroups.x * 256u)
        maximum = max(maximum, abs(source_buffer.data[index]));
    maxima[lane] = maximum;
    barrier();
    for (uint step = 128u; step > 0u; step >>= 1u) {
        if (lane < step)
            maxima[lane] = max(maxima[lane], maxima[lane + step]);
        barrier();
    }
    if (lane == 0u)
        scale_buffer.data[gl_WorkGroupID.x] =
            max(maxima[0] / parameters.divisor, 1.0e-8);
}
