#version 450
layout(local_size_x=8,local_size_y=8) in;
layout(binding=0) uniform sampler2D input_image;
layout(binding=1,std430) writeonly buffer O { float d[]; } o;
layout(push_constant) uniform P { uint source_width; uint source_height; uint width; uint height; } p;
void main() {
    uint x=gl_GlobalInvocationID.x, y=gl_GlobalInvocationID.y;
    if(x>=p.width || y>=p.height) return;
    // Match the host harness's floor-based nearest resize and raw RGB [0,1].
    // ZipDepth's Vulkan encoder weights already fold input normalization.
    ivec2 source=ivec2(x*p.source_width/p.width,y*p.source_height/p.height);
    vec3 c=texelFetch(input_image,source,0).rgb;
    uint plane=p.width*p.height,q=y*p.width+x;
    o.d[q]=c.r; o.d[plane+q]=c.g; o.d[2*plane+q]=c.b;
}
