#version 450
layout(local_size_x=8,local_size_y=8) in;
layout(binding=0,r32f) uniform writeonly image2D output_image;
layout(binding=1,std430) readonly buffer I{float d[];}i;
layout(push_constant) uniform P{uint iw;uint ih;uint ow;uint oh;}p;
void main(){uint x=gl_GlobalInvocationID.x,y=gl_GlobalInvocationID.y;if(x>=p.ow||y>=p.oh)return;float sx=p.ow>1?float(x)*(p.iw-1)/float(p.ow-1):0,sy=p.oh>1?float(y)*(p.ih-1)/float(p.oh-1):0;uint x0=uint(sx),y0=uint(sy),x1=min(x0+1,p.iw-1),y1=min(y0+1,p.ih-1);float v=mix(mix(i.d[y0*p.iw+x0],i.d[y0*p.iw+x1],sx-x0),mix(i.d[y1*p.iw+x0],i.d[y1*p.iw+x1],sx-x0),sy-y0);imageStore(output_image,ivec2(x,y),vec4(v));}
