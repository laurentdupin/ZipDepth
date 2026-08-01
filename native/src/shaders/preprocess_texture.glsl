#version 450
layout(local_size_x=8,local_size_y=8) in;
layout(binding=0) uniform sampler2D input_image;
layout(binding=1,std430) writeonly buffer O{float d[];}o;
layout(push_constant) uniform P{uint source_width;uint source_height;uint width;uint height;}p;
void main(){uint x=gl_GlobalInvocationID.x,y=gl_GlobalInvocationID.y;if(x>=p.width||y>=p.height)return;vec2 uv=(vec2(x,y)+.5)/vec2(p.width,p.height);vec3 c=texture(input_image,uv).rgb;uint plane=p.width*p.height,q=y*p.width+x;o.d[q]=(c.r-.485)/.229;o.d[plane+q]=(c.g-.456)/.224;o.d[2*plane+q]=(c.b-.406)/.225;}
